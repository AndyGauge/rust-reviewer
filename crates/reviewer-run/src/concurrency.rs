//! Adaptive concurrency for the critic stream.
//!
//! Rather than hardcode "one at a time" or "all at once", the harness *learns*
//! how many hunk-reviews to keep in flight by probing. It's congestion control
//! applied to inference requests: a gradient (Vegas-style) limiter with an AIMD
//! backstop.
//!
//! The signal is latency. Under a continuous-batching server, adding a request
//! to a not-yet-full batch costs ~nothing (it rides the same forward passes), so
//! latency stays flat while concurrency — and throughput — climb. Once the batch
//! saturates, further requests *queue*, and latency inflates without buying
//! throughput. That inflation is the knee. The limiter reads it directly:
//!
//! * `gradient = min_rtt / recent_rtt` — 1.0 when unsaturated, <1 when queueing.
//! * `new_limit = limit * gradient + sqrt(limit)` — grow while flat (the
//!   `sqrt` is the probe headroom), shrink when latency rises.
//! * an error (overload / 429 / OOM) triggers a multiplicative decrease — the
//!   AIMD safety backstop, since on this unified-memory box the hard ceiling is
//!   KV-cache, not latency.
//!
//! Known limitation (future work): `min_rtt` here is a plain rolling minimum, so
//! this doesn't yet re-probe upward under *non-stationary* load. Good enough to
//! find the knee within a run; a decaying `min_rtt` would let it track drift.

use std::time::Duration;

pub struct AdaptiveLimiter {
    limit: f64,
    min: f64,
    max: f64,
    /// Best (lowest) service time seen — the unsaturated latency floor.
    min_rtt: f64,
    last_rtt: f64,
    /// The limit after each update, for observability / plotting what it learned.
    trajectory: Vec<usize>,
}

impl AdaptiveLimiter {
    /// `max` is the OOM/KV-cache guard: exploration never climbs above it.
    pub fn new(min: usize, max: usize, start: usize) -> Self {
        let min = min.max(1) as f64;
        let max = (max as f64).max(min);
        Self {
            limit: (start as f64).clamp(min, max),
            min,
            max,
            min_rtt: f64::INFINITY,
            last_rtt: 0.0,
            trajectory: Vec::new(),
        }
    }

    /// Current concurrency target — how many reviews to keep in flight.
    pub fn limit(&self) -> usize {
        self.limit.floor().clamp(self.min, self.max) as usize
    }

    /// Record a completed review's latency and adjust (the Vegas gradient step).
    pub fn on_success(&mut self, rtt: Duration) {
        let s = rtt.as_secs_f64().max(1e-6);
        self.last_rtt = s;
        if s < self.min_rtt {
            self.min_rtt = s;
        }
        let gradient = (self.min_rtt / s).clamp(0.5, 1.0);
        let headroom = self.limit.sqrt();
        let new = self.limit * gradient + headroom;
        self.limit = new.clamp(self.min, self.max);
        self.trajectory.push(self.limit());
    }

    /// Record a failure (timeout / 429 / 503 / OOM) — multiplicative decrease.
    pub fn on_error(&mut self) {
        self.limit = (self.limit * 0.8).clamp(self.min, self.max);
        self.trajectory.push(self.limit());
    }

    pub fn settled(&self) -> usize {
        self.limit()
    }
    pub fn trajectory(&self) -> &[usize] {
        &self.trajectory
    }
    /// Unsaturated latency floor in milliseconds (what it learned as "fast").
    pub fn min_rtt_ms(&self) -> f64 {
        if self.min_rtt.is_finite() {
            self.min_rtt * 1000.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A server that batches freely up to `knee` in-flight requests, then queues:
    /// latency stays flat until saturation, then grows linearly. The limiter
    /// should discover a target near the knee without being told it.
    #[test]
    fn converges_near_the_saturation_knee() {
        let knee = 8usize;
        let base = 0.05; // 50 ms unsaturated service time
        let mut lim = AdaptiveLimiter::new(1, 32, 2);
        for _ in 0..800 {
            let n = lim.limit();
            let rtt = if n <= knee {
                base
            } else {
                base * n as f64 / knee as f64 // queueing beyond the batch
            };
            lim.on_success(Duration::from_secs_f64(rtt));
        }
        let s = lim.settled();
        assert!(
            s >= knee && s <= knee * 2,
            "settled at {s}, expected near knee {knee} (+ probe headroom)"
        );
    }

    #[test]
    fn error_triggers_multiplicative_backoff() {
        let mut lim = AdaptiveLimiter::new(1, 32, 16);
        let before = lim.settled();
        lim.on_error();
        assert!(lim.settled() < before, "error must shrink the limit");
    }

    #[test]
    fn never_exceeds_the_ceiling() {
        // Constant low latency ⇒ never saturates ⇒ wants to grow forever, but
        // the ceiling is the OOM guard.
        let mut lim = AdaptiveLimiter::new(1, 6, 2);
        for _ in 0..200 {
            lim.on_success(Duration::from_secs_f64(0.01));
        }
        assert_eq!(lim.settled(), 6);
    }

    #[test]
    fn does_not_get_stuck_at_the_floor() {
        let mut lim = AdaptiveLimiter::new(2, 32, 2);
        for _ in 0..50 {
            lim.on_success(Duration::from_secs_f64(0.02));
        }
        assert!(lim.settled() > 2, "should have probed upward from the start");
    }
}
