//! The app's files packed behind the setup program: `setup.exe = [this program][zstd stream][footer]`.
//!
//! The stream is a list of entries (`u16` path length, the path with `/` separators, `u64` size, the bytes) ended by
//! a zero length. The footer says where the stream starts, how long it is and how many bytes it unpacks to. The
//! program in front is the same one that runs the uninstall, so `uninstall.exe` is that part alone.
use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

const MAGIC: &[u8; 8] = b"DSMSETUP";
const FOOTER: u64 = 32;
/// zstd window: matches across the 470 MB cuBLAS library need a long one (the reader must allow the same)
const WINDOW_LOG: u32 = 27;

pub struct Payload {
    exe: PathBuf,
    /// bytes of the program in front of the stream
    pub stub: u64,
    len: u64,
    /// bytes of the unpacked files
    pub raw: u64,
}

impl Payload {
    /// The payload behind this program, or None when it has none (a development build, or `uninstall.exe`).
    pub fn of_this_program() -> Result<Option<Payload>> {
        Self::of(&std::env::current_exe()?)
    }

    pub fn of(exe: &Path) -> Result<Option<Payload>> {
        let mut f = File::open(exe)?;
        let size = f.metadata()?.len();
        if size < FOOTER {
            return Ok(None);
        }
        f.seek(SeekFrom::Start(size - FOOTER))?;
        let mut foot = [0u8; FOOTER as usize];
        f.read_exact(&mut foot)?;
        if &foot[..8] != MAGIC {
            return Ok(None);
        }
        let n = |i: usize| u64::from_le_bytes(foot[i..i + 8].try_into().unwrap());
        let (stub, len, raw) = (n(8), n(16), n(24));
        if stub + len + FOOTER != size {
            bail!("the installer is damaged (download it again)");
        }
        Ok(Some(Payload { exe: exe.to_path_buf(), stub, len, raw }))
    }

    /// Unpack every file under `dest`, reporting (bytes written so far, current file). Returns the files' paths
    /// relative to `dest`.
    pub fn extract(&self, dest: &Path, mut progress: impl FnMut(u64, &str)) -> Result<Vec<String>> {
        let mut f = File::open(&self.exe)?;
        f.seek(SeekFrom::Start(self.stub))?;
        let mut z = zstd::Decoder::new(BufReader::with_capacity(1 << 20, f.take(self.len)))?;
        z.window_log_max(WINDOW_LOG)?;
        let (mut files, mut done) = (vec![], 0u64);
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let mut n = [0u8; 2];
            z.read_exact(&mut n)?;
            let n = u16::from_le_bytes(n) as usize;
            if n == 0 {
                return Ok(files);
            }
            let mut name = vec![0u8; n];
            z.read_exact(&mut name)?;
            let name = String::from_utf8(name).context("bad file name in the installer")?;
            let mut size = [0u8; 8];
            z.read_exact(&mut size)?;
            let mut left = u64::from_le_bytes(size);
            let path = safe_join(dest, &name)?;
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let mut out = BufWriter::with_capacity(1 << 20, File::create(&path).with_context(|| format!("write {}", path.display()))?);
            while left > 0 {
                let k = buf.len().min(left as usize);
                z.read_exact(&mut buf[..k])?;
                out.write_all(&buf[..k])?;
                left -= k as u64;
                done += k as u64;
                progress(done, &name);
            }
            out.flush()?;
            files.push(name);
        }
    }

    /// Write this program without its payload (the uninstaller).
    pub fn write_stub(&self, to: &Path) -> Result<()> {
        let mut out = File::create(to)?;
        std::io::copy(&mut File::open(&self.exe)?.take(self.stub), &mut out)?;
        Ok(())
    }
}

/// `dest/name`, refusing names that would land outside `dest`.
fn safe_join(dest: &Path, name: &str) -> Result<PathBuf> {
    let rel = Path::new(name);
    if name.is_empty() || !rel.components().all(|c| matches!(c, Component::Normal(_))) {
        bail!("bad file name in the installer: {name}");
    }
    Ok(dest.join(rel))
}

/// Build `out` = this program + the files under `src` (the release's portable folder). Runs at packaging time.
pub fn pack(src: &Path, out: &Path) -> Result<()> {
    let me = std::env::current_exe()?;
    let stub = match Payload::of(&me)? {
        Some(p) => p.stub,
        None => std::fs::metadata(&me)?.len(),
    };
    let mut files = vec![];
    collect(src, src, &mut files)?;
    files.sort();
    let mut w = BufWriter::new(File::create(out)?);
    std::io::copy(&mut File::open(&me)?.take(stub), &mut w)?;
    let mut z = zstd::Encoder::new(w, 19)?;
    z.long_distance_matching(true)?;
    z.window_log(WINDOW_LOG)?;
    z.multithread(std::thread::available_parallelism().map_or(4, |n| n.get() as u32))?;
    let mut raw = 0u64;
    for rel in &files {
        let path = src.join(rel);
        let size = std::fs::metadata(&path)?.len();
        let name = rel.replace('\\', "/");
        z.write_all(&(name.len() as u16).to_le_bytes())?;
        z.write_all(name.as_bytes())?;
        z.write_all(&size.to_le_bytes())?;
        std::io::copy(&mut File::open(&path)?, &mut z)?;
        raw += size;
    }
    z.write_all(&0u16.to_le_bytes())?;
    let mut w = z.finish()?;
    w.flush()?;
    let mut f = w.into_inner().map_err(|e| e.into_error())?;
    let end = f.stream_position()?;
    f.write_all(MAGIC)?;
    for v in [stub, end - stub, raw] {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            collect(root, &p, out)?;
        } else {
            out.push(p.strip_prefix(root)?.to_string_lossy().to_string());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_outside_the_install_folder_are_refused() {
        let d = Path::new("C:/apps/Dotsmith");
        assert!(safe_join(d, "licenses/MIT.txt").is_ok());
        for bad in ["../evil.exe", "C:/Windows/evil.exe", "/evil", "", "a/../../b"] {
            assert!(safe_join(d, bad).is_err(), "{bad}");
        }
    }
}
