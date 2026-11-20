//! Branch delay compensation tests for `Net`.
//!
//! These tests verify that when parallel branches with different
//! processing latencies are mixed, enabling delay compensation aligns
//! the branches sample-accurately, that the reported latency and
//! frequency response describe the compensated sound, and that no
//! allocation happens in the audio thread once a graph is committed.

#![allow(clippy::float_cmp)]

use fundsp::prelude64::*;

/// Render `frames` mono samples from a network fed with an impulse at sample 0.
fn impulse_response(net: &mut Net, frames: usize) -> Vec<f32> {
    net.allocate();
    net.reset();
    let mut result = Vec::with_capacity(frames);
    for i in 0..frames {
        let input = if i == 0 { 1.0 } else { 0.0 };
        result.push(net.filter_mono(input));
    }
    result
}

/// Render `frames` mono samples from a network fed with an impulse at sample 0
/// using block processing.
fn impulse_response_block(net: &mut Net, frames: usize) -> Vec<f32> {
    net.allocate();
    net.reset();
    let mut result = vec![0.0f32; frames];
    let mut input_buffer = BufferVec::new(1);
    let mut output_buffer = BufferVec::new(1);
    let mut i = 0;
    while i < frames {
        let size = core::cmp::min(frames - i, MAX_BUFFER_SIZE);
        for j in 0..size {
            input_buffer.set_f32(0, j, if i + j == 0 { 1.0 } else { 0.0 });
        }
        net.process(
            size,
            &input_buffer.buffer_ref(),
            &mut output_buffer.buffer_mut(),
        );
        for j in 0..size {
            result[i + j] = output_buffer.at_f32(0, j);
        }
        i += size;
    }
    result
}

/// Index of the first sample whose magnitude exceeds `threshold`.
/// Averaging junctions halve amplitudes, so a unit impulse summed in a
/// two-branch `join` measures 0.5; the threshold stays below that.
fn first_nonzero(samples: &[f32], threshold: f32) -> Option<usize> {
    samples.iter().position(|&x| x.abs() > threshold)
}

/// Threshold used throughout: above the noise floor, below the 0.5
/// amplitude produced by a two-branch averaging junction.
const DETECT: f32 = 0.25;

/// Build a parallel network: dry branch (`pass`) summed with a wet branch
/// carrying an integer-sample `delay`. The summing junction is a `join` node.
fn dry_wet_net(delay_seconds: f32) -> (Net, NodeId, NodeId, NodeId) {
    let mut net = Net::new(1, 1);
    let dry = net.push(Box::new(pass()));
    let wet = net.push(Box::new(delay(delay_seconds)));
    let mix = net.push(Box::new(join::<U2>()));
    net.connect_input(0, dry, 0);
    net.connect_input(0, wet, 0);
    net.connect(dry, 0, mix, 0);
    net.connect(wet, 0, mix, 1);
    net.connect_output(mix, 0, 0);
    net.check();
    (net, dry, wet, mix)
}

#[test]
fn test_compensation_disabled_by_default() {
    let (net, _dry, _wet, _mix) = dry_wet_net(10.0 / DEFAULT_SR as f32);
    assert!(!net.delay_compensation());
}

#[test]
fn test_dry_wet_alignment_tick() {
    let delay_samples = 10usize;
    let (mut uncompensated, _d, _w, mix) = dry_wet_net(delay_samples as f32 / DEFAULT_SR as f32);
    let (mut compensated, _d2, _w2, mix2) = dry_wet_net(delay_samples as f32 / DEFAULT_SR as f32);
    compensated.set_delay_compensation(true);
    assert!(compensated.delay_compensation());

    // The dry input of the mix node must be delayed by the full wet latency.
    assert_eq!(compensated.input_compensation(mix2, 0), delay_samples);
    assert_eq!(compensated.input_compensation(mix2, 1), 0);
    // Disabled network applies no delay anywhere.
    assert_eq!(uncompensated.input_compensation(mix, 0), 0);

    let frames = 64;
    let raw = impulse_response(&mut uncompensated, frames);
    let aligned = impulse_response(&mut compensated, frames);

    // Without compensation the dry impulse arrives immediately.
    assert_eq!(first_nonzero(&raw, DETECT), Some(0));
    // With compensation both branches are aligned at the wet latency:
    // the first non-silent sample is the summed dry + wet impulse.
    assert_eq!(first_nonzero(&aligned, DETECT), Some(delay_samples));
    // The aligned impulse contains both branches (dry 1.0 plus wet 1.0).
    assert!((aligned[delay_samples] - 1.0).abs() < 1.0e-5);
    // Everything before the aligned point is silent.
    assert!(aligned[..delay_samples].iter().all(|&x| x == 0.0));
    // And nothing comes after the single aligned impulse.
    assert!(aligned[delay_samples + 1..].iter().all(|&x| x == 0.0));
}

#[test]
fn test_dry_wet_alignment_block() {
    let delay_samples = 37usize;
    let (mut compensated, _d, _w, mix) = dry_wet_net(delay_samples as f32 / DEFAULT_SR as f32);
    compensated.set_delay_compensation(true);
    assert_eq!(compensated.input_compensation(mix, 0), delay_samples);

    // A delay longer than the 64-sample block boundary exercises the ring
    // buffer across several blocks.
    let aligned = impulse_response_block(&mut compensated, 256);
    assert_eq!(first_nonzero(&aligned, DETECT), Some(delay_samples));
    assert!((aligned[delay_samples] - 1.0).abs() < 1.0e-5);
    assert!(aligned[..delay_samples].iter().all(|&x| x == 0.0));
    assert!(aligned[delay_samples + 1..].iter().all(|&x| x == 0.0));
}

#[test]
fn test_tick_and_block_paths_agree() {
    let delay_samples = 23usize;
    let (mut tick_net, _, _, _) = dry_wet_net(delay_samples as f32 / DEFAULT_SR as f32);
    let (mut block_net, _, _, _) = dry_wet_net(delay_samples as f32 / DEFAULT_SR as f32);
    tick_net.set_delay_compensation(true);
    block_net.set_delay_compensation(true);

    let via_tick = impulse_response(&mut tick_net, 200);
    let via_block = impulse_response_block(&mut block_net, 200);
    for (a, b) in via_tick.iter().zip(via_block.iter()) {
        assert!(
            (a - b).abs() < 1.0e-6,
            "tick/block mismatch: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn test_reported_latency_covers_compensation() {
    // The limiter reports a lookahead latency of 441 samples at 44.1 kHz.
    let limiter_latency = limiter(0.01, 0.01).latency().unwrap();
    assert_eq!(limiter_latency, 441.0);

    let mut net = Net::new(1, 1);
    let dry = net.push(Box::new(pass()));
    let wet = net.push(Box::new(limiter(0.01, 0.01)));
    let mix = net.push(Box::new(join::<U2>()));
    net.connect_input(0, dry, 0);
    net.connect_input(0, wet, 0);
    net.connect(dry, 0, mix, 0);
    net.connect(wet, 0, mix, 1);
    net.connect_output(mix, 0, 0);

    // Without compensation the fastest path (the dry branch) sets the
    // reported network latency, which is zero here.
    assert_eq!(net.latency(), Some(0.0));

    net.set_delay_compensation(true);
    net.allocate();

    // The dry branch gets delayed by the full limiter latency at the mix.
    assert_eq!(net.input_compensation(mix, 0), 441);
    assert_eq!(net.input_compensation(mix, 1), 0);
    // The network now reports the compensated latency.
    assert_eq!(net.latency(), Some(limiter_latency));

    // Offline rendering: discarding the reported pre-delay restores the
    // correct starting point. The dry impulse must land at sample 0 of
    // the trimmed output.
    let total = 1024;
    let mut rendered = impulse_response(&mut net, total + 441);
    let trim = net.latency().unwrap() as usize;
    rendered.drain(..trim);
    assert_eq!(first_nonzero(&rendered, DETECT), Some(0));
}

#[test]
fn test_frequency_response_phase_shift() {
    use num_complex::Complex64;

    // Two branches with different latencies feeding separate outputs.
    // Output compensation delays the fast output by 16 samples.
    let delay_samples = 16usize;
    let mut net = Net::new(2, 2);
    let fast = net.push(Box::new(pass()));
    let slow = net.push(Box::new(delay(delay_samples as f32 / DEFAULT_SR as f32)));
    net.connect_input(0, fast, 0);
    net.connect_input(1, slow, 0);
    net.connect_output(fast, 0, 0);
    net.connect_output(slow, 0, 1);
    net.set_delay_compensation(true);
    net.allocate();
    assert_eq!(net.output_compensation(0), delay_samples);
    assert_eq!(net.output_compensation(1), 0);

    let frequency = 1000.0;
    let expected_phase = -f64::TAU * frequency * delay_samples as f64 / DEFAULT_SR;

    // The fast output is pure unit gain delayed by the compensation:
    // magnitude one and the exact all-pass phase shift of the delay.
    let fast_response = net.response(0, frequency).unwrap();
    assert!((fast_response.norm() - 1.0).abs() < 1.0e-9);
    assert!((fast_response.arg() - expected_phase).abs() < 1.0e-9);
    assert_eq!(fast_response, Complex64::from_polar(1.0, expected_phase));

    // The slow output keeps its own physical delay response, which has
    // the same phase at this frequency but is unaffected by extra output
    // compensation.
    let slow_response = net.response(1, frequency).unwrap();
    assert!((slow_response.norm() - 1.0).abs() < 1.0e-9);
    assert!((slow_response.arg() - expected_phase).abs() < 1.0e-9);

    // Turning compensation off removes the added phase from the fast output.
    net.set_delay_compensation(false);
    let raw_response = net.response(0, frequency).unwrap();
    assert!((raw_response.arg() - 0.0).abs() < 1.0e-9);
    assert!((raw_response.norm() - 1.0).abs() < 1.0e-9);
}

#[test]
fn test_global_output_alignment() {
    // Two independent branches with different latencies feeding two
    // global outputs. Output compensation aligns them with each other.
    let mut net = Net::new(2, 2);
    let fast = net.push(Box::new(pass()));
    let slow = net.push(Box::new(delay(8.0 / DEFAULT_SR as f32)));
    net.connect_input(0, fast, 0);
    net.connect_input(1, slow, 0);
    net.connect_output(fast, 0, 0);
    net.connect_output(slow, 0, 1);
    net.set_delay_compensation(true);
    net.check();

    assert_eq!(net.output_compensation(0), 8);
    assert_eq!(net.output_compensation(1), 0);

    net.allocate();
    net.reset();
    let mut out0_first = None;
    let mut out1_first = None;
    for i in 0..64 {
        // Impulse on both inputs simultaneously.
        let impulse = if i == 0 { 1.0 } else { 0.0 };
        let (x, y) = net.filter_stereo(impulse, impulse);
        if out0_first.is_none() && x.abs() > DETECT {
            out0_first = Some(i);
        }
        if out1_first.is_none() && y.abs() > DETECT {
            out1_first = Some(i);
        }
    }
    // Both global outputs start at the same sample.
    assert_eq!(out0_first, Some(8));
    assert_eq!(out1_first, Some(8));
}

#[test]
fn test_disabling_compensation_restores_mixing() {
    let delay_samples = 5usize;
    let (mut net, _d, _w, mix) = dry_wet_net(delay_samples as f32 / DEFAULT_SR as f32);
    net.set_delay_compensation(true);
    net.allocate();
    assert_eq!(net.input_compensation(mix, 0), delay_samples);

    net.set_delay_compensation(false);
    assert!(!net.delay_compensation());
    assert_eq!(net.input_compensation(mix, 0), 0);
    assert_eq!(net.latency(), Some(0.0));

    let raw = impulse_response(&mut net, 32);
    // Original behavior: the dry impulse is at the first sample.
    assert_eq!(first_nonzero(&raw, DETECT), Some(0));
}

#[test]
fn test_multichannel_phase_preserved() {
    // A stereo branch with identical delay on both channels must keep
    // the channels sample-aligned with each other (same compensation).
    let mut net = Net::new(2, 2);
    let dry = net.push(Box::new(multipass::<U2>()));
    let wet = net.push(Box::new(multitick::<U2>() >> multitick::<U2>()));
    // MultiJoin<2, 2> expects 4 inputs: dry L, dry R, wet L, wet R.
    let mix = net.push(Box::new(multijoin::<U2, U2>()));
    net.connect_input(0, dry, 0);
    net.connect_input(1, dry, 1);
    net.connect_input(0, wet, 0);
    net.connect_input(1, wet, 1);
    net.connect(dry, 0, mix, 0);
    net.connect(dry, 1, mix, 1);
    net.connect(wet, 0, mix, 2);
    net.connect(wet, 1, mix, 3);
    net.connect_output(mix, 0, 0);
    net.connect_output(mix, 1, 1);
    net.set_delay_compensation(true);
    net.check();
    net.allocate();

    // The wet branch delays by two samples on each channel; each dry
    // input of the mix gets the same two-sample compensation.
    assert_eq!(net.input_compensation(mix, 0), 2);
    assert_eq!(net.input_compensation(mix, 1), 2);
    assert_eq!(net.input_compensation(mix, 2), 0);
    assert_eq!(net.input_compensation(mix, 3), 0);

    net.reset();
    // Impulse on both channels simultaneously.
    let mut left_first = None;
    let mut right_first = None;
    for i in 0..32 {
        let (l, r) = net.filter_stereo(
            if i == 0 { 1.0 } else { 0.0 },
            if i == 0 { 1.0 } else { 0.0 },
        );
        if left_first.is_none() && l.abs() > DETECT {
            left_first = Some(i);
        }
        if right_first.is_none() && r.abs() > DETECT {
            right_first = Some(i);
        }
    }
    assert_eq!(left_first, Some(2));
    assert_eq!(right_first, Some(2));
}

#[test]
fn test_recompensation_after_topology_edit() {
    // Start with a small wet delay, then replace it with a larger one and
    // verify the compensation updates to match the new effective graph.
    let (mut net, _dry, wet, mix) = dry_wet_net(4.0 / DEFAULT_SR as f32);
    net.set_delay_compensation(true);
    net.allocate();
    assert_eq!(net.input_compensation(mix, 0), 4);

    net.replace(wet, Box::new(delay(12.0 / DEFAULT_SR as f32)));
    // Recompute topology and grow the compensation buffers off-line.
    net.allocate();
    assert_eq!(net.input_compensation(mix, 0), 12);

    let aligned = impulse_response(&mut net, 64);
    assert_eq!(first_nonzero(&aligned, DETECT), Some(12));
    assert!((aligned[12] - 1.0).abs() < 1.0e-5);
}

#[test]
fn test_backend_commit_updates_compensation_without_audio_allocation() {
    use assert_no_alloc::*;

    let mut net = Net::new(1, 1);
    let dry = net.push(Box::new(pass()));
    let wet = net.push(Box::new(pass()));
    let mix = net.push(Box::new(join::<U2>()));
    net.connect_input(0, dry, 0);
    net.connect_input(0, wet, 0);
    net.connect(dry, 0, mix, 0);
    net.connect(wet, 0, mix, 1);
    net.connect_output(mix, 0, 0);
    net.set_delay_compensation(true);
    net.allocate();

    let mut backend = net.backend();
    // Warm up the backend so lazy initialization is done already.
    for i in 0..128 {
        let _ = backend.filter_mono(if i == 0 { 1.0 } else { 0.0 });
    }

    // Edit the frontend: give the wet branch a 7-sample delay.
    net.replace(wet, Box::new(delay(7.0 / DEFAULT_SR as f32)));
    net.allocate();
    net.commit();

    // Processing the committed graph on the audio thread must not
    // allocate or deallocate, even though the compensation changed.
    let mut out = Vec::new();
    assert_no_alloc(|| {
        for i in 0..128 {
            out.push(backend.filter_mono(if i == 0 { 1.0 } else { 0.0 }));
        }
    });

    // Both branches now arrive together at sample 7.
    assert_eq!(first_nonzero(&out, DETECT), Some(7));
    assert!((out[7] - 1.0).abs() < 1.0e-5);
}

#[test]
fn test_backend_toggle_compensation() {
    let mut net = Net::new(1, 1);
    let dry = net.push(Box::new(pass()));
    let wet = net.push(Box::new(delay(6.0 / DEFAULT_SR as f32)));
    let mix = net.push(Box::new(join::<U2>()));
    net.connect_input(0, dry, 0);
    net.connect_input(0, wet, 0);
    net.connect(dry, 0, mix, 0);
    net.connect(wet, 0, mix, 1);
    net.connect_output(mix, 0, 0);
    let mut backend = net.backend();

    // Initially uncompensated: dry impulse at sample 0.
    let mut before = Vec::new();
    for i in 0..32 {
        before.push(backend.filter_mono(if i == 0 { 1.0 } else { 0.0 }));
    }
    assert_eq!(first_nonzero(&before, DETECT), Some(0));

    // Enable compensation on the live frontend and commit.
    net.set_delay_compensation(true);
    net.allocate();
    net.commit();

    let mut after = Vec::new();
    for i in 0..32 {
        after.push(backend.filter_mono(if i == 0 { 1.0 } else { 0.0 }));
    }
    assert_eq!(first_nonzero(&after, DETECT), Some(6));
}

#[test]
fn test_compensation_with_bus_combinator() {
    // The `&` (bus) combinator merges two parallel networks and inserts
    // an internal summing junction. Compensation must follow the merged
    // vertices and edges.
    let dry = Net::wrap(Box::new(pass()));
    let wet = Net::wrap(Box::new(delay(11.0 / DEFAULT_SR as f32)));
    let mut net = dry & wet;
    net.set_delay_compensation(true);
    net.check();
    net.allocate();
    net.reset();

    let mut first = None;
    for i in 0..40 {
        let x = net.filter_mono(if i == 0 { 1.0 } else { 0.0 });
        if first.is_none() && x.abs() > DETECT {
            first = Some(i);
        }
    }
    assert_eq!(first, Some(11));
}

#[test]
fn test_compensation_with_sum_combinator() {
    // The `+` (sum) combinator joins two branches into one summed output.
    let a = Net::wrap(Box::new(pass()));
    let b = Net::wrap(Box::new(delay(7.0 / DEFAULT_SR as f32)));
    let mut net = a + b; // 2 inputs, 1 output
    net.set_delay_compensation(true);
    net.check();
    net.allocate();
    net.reset();

    let mut first = None;
    for i in 0..30 {
        let impulse = if i == 0 { 1.0 } else { 0.0 };
        let mut out = [0.0f32; 1];
        net.tick(&[impulse, impulse], &mut out);
        if first.is_none() && out[0].abs() > DETECT {
            first = Some(i);
        }
    }
    assert_eq!(first, Some(7));
}
