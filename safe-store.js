// Halo Go "safe store": sits between WebLLM and the iPhone's private file storage (OPFS).
// Safari sometimes refuses to save big downloads ("UnknownError … transient reason").
// Instead of failing, any file that can't be saved is kept in memory for this session,
// so the brain still loads. It just downloads again next launch.
// A brain file kept in memory is let go as soon as it's been copied to the graphics chip.

const mem = new Map();          // path -> Blob, for files Safari wouldn't save
export const status = { diskOk: true, firstError: "" };

const notFound = (what) => new DOMException(`${what} not found`, "NotFoundError");
const errName = (e) => (e && typeof e === "object" && "name" in e ? e.name : "");
const errText = (e) => `${errName(e) || "Error"}: ${(e && e.message) || e}`;

function diskFailed(e) {
  if (status.diskOk) { status.diskOk = false; status.firstError = errText(e); }
}

async function writeDisk(real, blob) {
  const CH = 4 * 1024 * 1024;
  // Worker-only sync handle first (writes in place, least memory), then the async writer
  if (typeof real.createSyncAccessHandle === "function") {
    let h = null;
    try {
      h = await real.createSyncAccessHandle();
      h.truncate(0);
      for (let o = 0; o < blob.size; o += CH) {
        const buf = new Uint8Array(await blob.slice(o, o + CH).arrayBuffer());
        let done = 0;
        while (done < buf.byteLength) {
          const n = h.write(done ? buf.subarray(done) : buf, { at: o + done });
          if (!n) throw new DOMException("Wrote nothing", "UnknownError");
          done += n;
        }
      }
      h.flush();
      if (h.getSize() !== blob.size) throw new DOMException("Saved size doesn't match", "UnknownError");
      return;
    } catch (e) {
      if (typeof real.createWritable !== "function") throw e;
    } finally {
      try { h && h.close(); } catch {}
    }
  }
  const w = await real.createWritable();
  try { await w.write(blob); await w.close(); }
  catch (e) { try { await w.abort(); } catch {} throw e; }
}

class SafeFile {
  constructor(dir, name, real) { this.kind = "file"; this.name = name; this.dir = dir; this.path = dir.path + "/" + name; this.real = real; }
  async getFile() {
    const b = mem.get(this.path);
    if (b) {
      if (!/\.bin$/.test(this.name)) return b;
      // Brain pieces are read once (to copy them to the GPU), so free them right after
      const path = this.path, copy = b.slice(0, b.size, b.type);
      copy.arrayBuffer = () => { mem.delete(path); return Blob.prototype.arrayBuffer.call(copy); };
      return copy;
    }
    if (!this.real) throw notFound(this.name);
    try { return await this.real.getFile(); }
    catch (e) {
      if (errName(e) === "NotFoundError" || errName(e) === "InvalidStateError") throw e;
      throw notFound(this.name);  // unreadable on disk: treat as not downloaded yet
    }
  }
  async createWritable() {
    let parts = [];
    const file = this;
    return {
      async write(v) {
        if (!parts) throw new DOMException("Writer closed", "InvalidStateError");
        if (v && typeof v === "object" && "data" in v && !(v instanceof Blob) && !ArrayBuffer.isView(v)) v = v.data;
        parts.push(ArrayBuffer.isView(v) ? v.slice() : v);   // keep our own copy
      },
      async close() {
        const blob = new Blob(parts); parts = null;
        if (status.diskOk) {
          try {
            if (!file.real) throw file.dir.err || new DOMException("No folder on disk", "UnknownError");
            await writeDisk(file.real, blob);
            mem.delete(file.path);
            return;
          } catch (e) {
            diskFailed(e);
            try { await file.dir.real?.removeEntry(file.name); } catch {}
          }
        }
        mem.set(file.path, blob);
      },
      async abort() { parts = null; },
    };
  }
}

class SafeDir {
  constructor(path, real, err) { this.kind = "directory"; this.path = path; this.real = real; this.err = err; this.name = path.split("/").pop(); }
  async getDirectoryHandle(name, opts = {}) {
    const path = this.path + "/" + name;
    if (this.real) {
      try { return new SafeDir(path, await this.real.getDirectoryHandle(name, opts)); }
      catch (e) {
        if (errName(e) === "NotFoundError" && !opts.create && !hasMemUnder(path)) throw e;
        diskFailed(e);
        return new SafeDir(path, null, e);
      }
    }
    if (!opts.create && !hasMemUnder(path)) throw notFound(name);
    return new SafeDir(path, null, this.err);
  }
  async getFileHandle(name, opts = {}) {
    const path = this.path + "/" + name;
    if (mem.has(path)) {
      let real = null;
      try { real = this.real && status.diskOk ? await this.real.getFileHandle(name, { create: true }) : null; } catch {}
      return new SafeFile(this, name, real);
    }
    if (this.real) {
      try { return new SafeFile(this, name, await this.real.getFileHandle(name, opts)); }
      catch (e) {
        if (errName(e) === "NotFoundError" || errName(e) === "TypeMismatchError") { if (!opts.create) throw notFound(name); }
        if (!opts.create) throw notFound(name);
        diskFailed(e);
        return new SafeFile(this, name, null);
      }
    }
    if (!opts.create) throw notFound(name);
    return new SafeFile(this, name, null);
  }
  async removeEntry(name, opts) {
    const path = this.path + "/" + name;
    let had = mem.delete(path);
    for (const k of [...mem.keys()]) if (k.startsWith(path + "/")) { mem.delete(k); had = true; }
    if (this.real) {
      try { await this.real.removeEntry(name, opts); return; }
      catch (e) { if (had) return; if (errName(e) === "NotFoundError") throw e; return; }
    }
    if (!had) throw notFound(name);
  }
}
const hasMemUnder = (path) => { for (const k of mem.keys()) if (k.startsWith(path + "/")) return true; return false; };

export function installSafeStore(nav = globalThis.navigator) {
  const storage = nav && nav.storage;
  const realGet = storage && typeof storage.getDirectory === "function" ? storage.getDirectory.bind(storage) : null;
  const getDirectory = async () => {
    if (!realGet) return new SafeDir("", null, new DOMException("No file storage", "UnknownError"));
    try { return new SafeDir("", await realGet()); }
    catch (e) { diskFailed(e); return new SafeDir("", null, e); }
  };
  if (storage) Object.defineProperty(storage, "getDirectory", { value: getDirectory, configurable: true, writable: true });
  else Object.defineProperty(nav, "storage", { value: { getDirectory }, configurable: true });
}

export const _test = { mem, SafeDir };
