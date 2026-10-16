//! SHA-256 (FIPS 180-4), dependency-free, for the streamed output identity.
//!
//! Why it is here and not a crate: this writer is dependency-free on
//! purpose (see `Cargo.toml`), and the product tape's identity must equal
//! the SHA-256 Python's `hashlib` computes over the same bytes, which is
//! exactly what the reference algorithm gives.  Two compression paths:
//! the portable one, and the x86-64 SHA extensions (`sha_ni`) chosen at
//! RUN time when the CPU has them -- the same instructions OpenSSL uses,
//! so hashing while writing costs what hashing a re-read costs, without
//! the re-read.  Both are checked against the FIPS vectors and against
//! each other in the tests below.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// A running SHA-256.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total: u64,
    accelerated: bool,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Sha256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sha256")
            .field("total", &self.total)
            .field("accelerated", &self.accelerated)
            .finish()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Sha256 {
            state: H0,
            buf: [0; 64],
            buf_len: 0,
            total: 0,
            accelerated: shani_available(),
        }
    }

    /// The portable compression only -- for the cross-check tests.
    pub fn new_portable() -> Self {
        Sha256 {
            accelerated: false,
            ..Self::new()
        }
    }

    /// Bytes hashed so far.
    pub fn len(&self) -> u64 {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len < 64 {
                return;
            }
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }
        let whole = data.len() / 64 * 64;
        if whole > 0 {
            self.compress(&data[..whole]);
        }
        let rest = &data[whole..];
        self.buf[..rest.len()].copy_from_slice(rest);
        self.buf_len = rest.len();
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total.wrapping_mul(8);
        let mut pad = [0u8; 72];
        pad[0] = 0x80;
        let pad_len = if self.buf_len < 56 { 56 - self.buf_len } else { 120 - self.buf_len };
        pad[pad_len..pad_len + 8].copy_from_slice(&bit_len.to_be_bytes());
        // `update` would count the padding into `total`; feed it raw.
        let total = self.total;
        self.update(&pad[..pad_len + 8]);
        self.total = total;
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 32];
        for (chunk, word) in out.chunks_exact_mut(4).zip(self.state.iter()) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// `blocks.len()` is a multiple of 64.
    fn compress(&mut self, blocks: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            if self.accelerated {
                // SAFETY: `accelerated` is set only when the CPU reported
                // every feature `compress_shani` is compiled for.
                unsafe { compress_shani(&mut self.state, blocks) };
                return;
            }
        }
        compress_portable(&mut self.state, blocks);
    }
}

/// Hex spelling, as `hashlib.hexdigest()` gives it.
pub fn hex(digest: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

fn shani_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("sha")
            && std::arch::is_x86_feature_detected!("sse2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn compress_portable(state: &mut [u32; 8], blocks: &[u8]) {
    let mut w = [0u32; 64];
    for block in blocks.chunks_exact(64) {
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
}

/// The SHA-NI compression (Intel's published instruction sequence).
///
/// # Safety
/// The CPU must support `sha`, `sse2`, `ssse3` and `sse4.1`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
#[allow(unused_unsafe)]
unsafe fn compress_shani(state: &mut [u32; 8], blocks: &[u8]) {
    use std::arch::x86_64::*;

    unsafe {
        let mask = _mm_set_epi64x(0x0C0D_0E0F_0809_0A0Bu64 as i64, 0x0405_0607_0001_0203u64 as i64);
        let state_ptr = state.as_ptr() as *const __m128i;
        let dcba = _mm_loadu_si128(state_ptr);
        let efgh = _mm_loadu_si128(state_ptr.add(1));
        let cdab = _mm_shuffle_epi32(dcba, 0xB1);
        let efgh = _mm_shuffle_epi32(efgh, 0x1B);
        let mut abef = _mm_alignr_epi8(cdab, efgh, 8);
        let mut cdgh = _mm_blend_epi16(efgh, cdab, 0xF0);
        let k_ptr = K.as_ptr() as *const __m128i;

        for block in blocks.chunks_exact(64) {
            let abef_save = abef;
            let cdgh_save = cdgh;
            let data = block.as_ptr() as *const __m128i;
            let mut w = [
                _mm_shuffle_epi8(_mm_loadu_si128(data), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(data.add(1)), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(data.add(2)), mask),
                _mm_shuffle_epi8(_mm_loadu_si128(data.add(3)), mask),
            ];
            for i in 0..16 {
                let words = if i < 4 {
                    w[i]
                } else {
                    let t1 = _mm_sha256msg1_epu32(w[i % 4], w[(i + 1) % 4]);
                    let t2 = _mm_alignr_epi8(w[(i + 3) % 4], w[(i + 2) % 4], 4);
                    let next = _mm_sha256msg2_epu32(_mm_add_epi32(t1, t2), w[(i + 3) % 4]);
                    w[i % 4] = next;
                    next
                };
                let k = _mm_loadu_si128(k_ptr.add(i));
                let t1 = _mm_add_epi32(words, k);
                cdgh = _mm_sha256rnds2_epu32(cdgh, abef, t1);
                let t2 = _mm_shuffle_epi32(t1, 0x0E);
                abef = _mm_sha256rnds2_epu32(abef, cdgh, t2);
            }
            abef = _mm_add_epi32(abef, abef_save);
            cdgh = _mm_add_epi32(cdgh, cdgh_save);
        }

        let feba = _mm_shuffle_epi32(abef, 0x1B);
        let dchg = _mm_shuffle_epi32(cdgh, 0xB1);
        let dcba = _mm_blend_epi16(feba, dchg, 0xF0);
        let hgef = _mm_alignr_epi8(dchg, feba, 8);
        let out = state.as_mut_ptr() as *mut __m128i;
        _mm_storeu_si128(out, dcba);
        _mm_storeu_si128(out.add(1), hgef);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(hasher: Sha256, data: &[u8]) -> String {
        let mut hasher = hasher;
        hasher.update(data);
        hex(&hasher.finalize())
    }

    #[test]
    fn fips_vectors_both_paths() {
        let cases: [(&[u8], &str); 3] = [
            (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
        ];
        for (data, want) in cases {
            assert_eq!(digest(Sha256::new_portable(), data), want);
            assert_eq!(digest(Sha256::new(), data), want);
        }
        let million_a = vec![b'a'; 1_000_000];
        let want = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
        assert_eq!(digest(Sha256::new_portable(), &million_a), want);
        assert_eq!(digest(Sha256::new(), &million_a), want);
    }

    #[test]
    fn accelerated_and_portable_agree_under_ragged_updates() {
        let data: Vec<u8> = (0..200_003u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let mut fast = Sha256::new();
        let mut slow = Sha256::new_portable();
        let mut offset = 0usize;
        let mut step = 1usize;
        while offset < data.len() {
            let end = (offset + step).min(data.len());
            fast.update(&data[offset..end]);
            slow.update(&data[offset..end]);
            offset = end;
            step = step * 3 % 977 + 1;
        }
        assert_eq!(fast.len(), data.len() as u64);
        assert_eq!(hex(&fast.finalize()), hex(&slow.finalize()));
    }
}
