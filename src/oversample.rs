//! Oversampling.

use super::audionode::*;
use super::buffer::*;
use super::graph::*;
use super::math::*;
use super::signal::*;
use super::*;
use numeric_array::typenum::*;

#[inline]
fn interpolating_filter(input_buffer: &[f32], new_sample_index: usize) -> (f32, f32) {
    let start_sample = new_sample_index + START_SAMPLE_OFFSET_HALF;
    let mut acc_even = f32x8::splat(0.);
    let mut acc_odd = f32x8::splat(0.);

    for (i, (even_coeff_slice, odd_coeff_slice)) in INTERPOLATING_EVEN_COEFFS
        .iter()
        .copied()
        .zip(INTERPOLATING_ODD_COEFFS.iter().copied())
        .enumerate()
    {
        let loop_start = start_sample + i * 8;
        let samples = f32x8::from([
            input_buffer[(loop_start) & 0x7f],
            input_buffer[(loop_start + 1) & 0x7f],
            input_buffer[(loop_start + 2) & 0x7f],
            input_buffer[(loop_start + 3) & 0x7f],
            input_buffer[(loop_start + 4) & 0x7f],
            input_buffer[(loop_start + 5) & 0x7f],
            input_buffer[(loop_start + 6) & 0x7f],
            input_buffer[(loop_start + 7) & 0x7f],
        ]);
        acc_even = samples.mul_add(f32x8::from(even_coeff_slice), acc_even);
        acc_odd = samples.mul_add(f32x8::from(odd_coeff_slice), acc_odd);
    }

    // multiply output by 2 since we only summed half the sample * coefficient products!
    (acc_even.reduce_add() * 2.0, acc_odd.reduce_add() * 2.0)
}

#[inline]
fn decimating_filter(output_buffer: &[f32], last_sample_index: usize) -> f32 {
    let start_sample = last_sample_index + START_SAMPLE_OFFSET_FULL;
    let mut accumulator = f32x8::splat(0.);

    for (i, coeffs_slice) in DECIMATING_COEFFS.iter().copied().enumerate() {
        let loop_start = start_sample + i * 8;
        let samples = f32x8::from([
            output_buffer[(loop_start) & 0x7f],
            output_buffer[(loop_start + 1) & 0x7f],
            output_buffer[(loop_start + 2) & 0x7f],
            output_buffer[(loop_start + 3) & 0x7f],
            output_buffer[(loop_start + 4) & 0x7f],
            output_buffer[(loop_start + 5) & 0x7f],
            output_buffer[(loop_start + 6) & 0x7f],
            output_buffer[(loop_start + 7) & 0x7f],
        ]);
        accumulator = samples.mul_add(f32x8::from(coeffs_slice), accumulator);
    }
    accumulator.reduce_add()
}

#[derive(Clone)]
pub struct Oversampler<X>
where
    X: AudioNode,
    X::Inputs: Size<f32>,
    X::Outputs: Size<f32>,
    X::Inputs: Size<Frame<f32, U128>>,
    X::Outputs: Size<Frame<f32, U128>>,
{
    x: X,
    inv: Frame<Frame<f32, U128>, X::Inputs>,
    outv: Frame<Frame<f32, U128>, X::Outputs>,
    input_rb_index: usize,
    output_rb_index: usize,
    sample_rate: f64,
}

/// Group delay of the interpolation and decimation filters in outer samples.
/// The interpolation and decimation stages each contribute the group delay of the
/// halfband filter at DC, which is the delay of the whole oversampling layer
/// (the inner node's own latency is added on top of this in `route`).
fn filter_latency() -> f64 {
    let mut even_sum = 0.0;
    let mut odd_sum = 0.0;
    let mut even_delay = 0.0;
    let mut odd_delay = 0.0;
    for d in 0..=HALFBAND_MIN_LEN / 2 {
        let w = HALFBAND_MIN[d * 2] as f64;
        even_sum += w;
        even_delay += (HALFBAND_MIN_LEN / 2 - d) as f64 * w;
    }
    for t in 0..HALFBAND_MIN_LEN / 2 {
        let w = HALFBAND_MIN[t * 2 + 1] as f64;
        odd_sum += w;
        odd_delay += (HALFBAND_MIN_LEN / 2 - 1 - t) as f64 * w;
    }
    even_delay / even_sum + odd_delay / odd_sum
}

/// Frequency response of the interpolation and decimation filters at `omega`,
/// the outer sample rate angular frequency (radians per outer sample).
/// The interpolation stage produces even and odd inner samples with responses
/// `even` and `odd`, and the cascade with the decimation stage has response
/// `even * odd` at the outer rate.
fn filter_response(omega: f64) -> Complex64 {
    let mut even = Complex64::new(0.0, 0.0);
    for d in 0..=HALFBAND_MIN_LEN / 2 {
        let phase = -omega * (HALFBAND_MIN_LEN / 2 - d) as f64;
        even += Complex64::from_polar(2.0 * HALFBAND_MIN[d * 2] as f64, phase);
    }
    let mut odd = Complex64::new(0.0, 0.0);
    for t in 0..HALFBAND_MIN_LEN / 2 {
        let phase = -omega * (HALFBAND_MIN_LEN / 2 - 1 - t) as f64;
        odd += Complex64::from_polar(2.0 * HALFBAND_MIN[t * 2 + 1] as f64, phase);
    }
    even * odd
}

impl<X> Oversampler<X>
where
    X: AudioNode,
    X::Inputs: Size<f32>,
    X::Outputs: Size<f32>,
    X::Inputs: Size<Frame<f32, U128>>,
    X::Outputs: Size<Frame<f32, U128>>,
{
    /// Create new oversampler. 2x oversamples enclosed node.
    pub fn new(sample_rate: f64, mut node: X) -> Self {
        let inner_sr = sample_rate * 2.0;
        node.set_sample_rate(inner_sr);
        let hash = node.ping(true, AttoHash::new(Self::ID));
        node.ping(false, hash);
        Self {
            x: node,
            inv: Frame::default(),
            outv: Frame::default(),
            input_rb_index: 0,
            output_rb_index: 0,
            sample_rate,
        }
    }

    // Access enclosed node.
    pub fn node(&self) -> &X {
        &self.x
    }

    // Access enclosed node.
    pub fn node_mut(&mut self) -> &mut X {
        &mut self.x
    }
}

impl<X> AudioNode for Oversampler<X>
where
    X: AudioNode,
    X::Inputs: Size<f32>,
    X::Outputs: Size<f32>,
    X::Inputs: Size<Frame<f32, U128>>,
    X::Outputs: Size<Frame<f32, U128>>,
{
    const ID: u64 = 51;
    type Inputs = X::Inputs;
    type Outputs = X::Outputs;

    fn reset(&mut self) {
        self.x.reset();
        self.inv = Frame::default();
        self.outv = Frame::default();
        self.input_rb_index = 0;
        self.output_rb_index = 0;
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        if self.sample_rate == sample_rate {
            // Sample rate is unchanged, so maintain current state.
            return;
        }
        self.sample_rate = sample_rate;
        let inner_sr = sample_rate * 2.0;
        self.x.set_sample_rate(inner_sr);
        // The delay of the filter chain is defined in outer samples and does not depend on
        // the sample rate. Clear the filter history so that the reported timeline is exact
        // from the next sample on, regardless of the previous sample rate.
        self.inv = Frame::default();
        self.outv = Frame::default();
        self.input_rb_index = 0;
        self.output_rb_index = 0;
    }

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut over_input = Frame::default();
        let mut over_input2 = Frame::default();

        // add input sample to input ringbuf
        for channel in 0..Self::Inputs::USIZE {
            self.inv[channel][self.input_rb_index] = input[channel];

            let (even, odd) = interpolating_filter(&self.inv[channel], self.input_rb_index);
            over_input[channel] = even;
            over_input2[channel] = odd;
        }
        self.input_rb_index = (self.input_rb_index + 1) & 0x7f;

        // get output and push to output ringbuf
        let over_output = self.x.tick(&over_input);
        for channel in 0..Self::Outputs::USIZE {
            self.outv[channel][self.output_rb_index] = over_output[channel];
        }
        self.output_rb_index = (self.output_rb_index + 1) & 0x7f;

        let over_output2 = self.x.tick(&over_input2);
        for channel in 0..Self::Outputs::USIZE {
            self.outv[channel][self.output_rb_index] = over_output2[channel];
        }

        let output: Frame<f32, Self::Outputs> =
            Frame::generate(|channel| decimating_filter(&self.outv[channel], self.output_rb_index));

        self.output_rb_index = (self.output_rb_index + 1) & 0x7f;

        output
    }

    #[inline]
    fn process(&mut self, size: usize, input: &BufferRef, output: &mut BufferMut) {
        let mut inner_input: BufferArray<Self::Inputs> = BufferArray::new();
        let mut inner_output: BufferArray<Self::Outputs> = BufferArray::new();

        // Process in chunks of up to half a buffer so that the interpolated inner block
        // fits in a buffer (2 * 32 = 64 samples). Any block length is supported, including
        // odd lengths: every input sample is interpolated and every output sample is written.
        let mut offset = 0;
        while offset < size {
            let chunk = min(size - offset, MAX_BUFFER_SIZE / 2);

            // Interpolate `chunk` input samples into twice as many inner samples.
            for i in 0..chunk {
                for channel in 0..Self::Inputs::USIZE {
                    self.inv[channel][self.input_rb_index] = input.at_f32(channel, i + offset);

                    let (even, odd) = interpolating_filter(&self.inv[channel], self.input_rb_index);
                    inner_input.set_f32(channel, i * 2, even);
                    inner_input.set_f32(channel, i * 2 + 1, odd);
                }

                self.input_rb_index = (self.input_rb_index + 1) & 0x7f;
            }

            // The inner node advances at twice the outer sample rate, so it processes
            // exactly the 2 * `chunk` samples that were interpolated above.
            self.x.process(
                chunk * 2,
                &(inner_input.buffer_ref()),
                &mut inner_output.buffer_mut(),
            );

            // Decimate into output buffer.
            for i in 0..chunk {
                for channel in 0..Self::Outputs::USIZE {
                    self.outv[channel][self.output_rb_index] = inner_output.at_f32(channel, i * 2);
                    let next_output_index = (self.output_rb_index + 1) & 0x7f;
                    self.outv[channel][next_output_index] = inner_output.at_f32(channel, i * 2 + 1);

                    let output_sample = decimating_filter(&self.outv[channel], next_output_index);
                    output.set_f32(channel, i + offset, output_sample);
                }

                self.output_rb_index = (self.output_rb_index + 2) & 0x7f;
            }

            offset += chunk;
        }
    }

    fn route(&mut self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        // The inner node runs at twice the outer sample rate. It reports latency in inner
        // samples, which are half the length of outer samples. The interpolation and
        // decimation filters add their own group delay and magnitude/phase response.
        let omega = if self.sample_rate > 0.0 {
            f64::TAU * frequency / self.sample_rate
        } else {
            0.0
        };
        let extra_latency = filter_latency();
        let extra_response = filter_response(omega);
        let inner = self.x.route(input, frequency);
        let mut output = SignalFrame::new(self.outputs());
        for i in 0..self.outputs() {
            output.set(
                i,
                match inner.at(i) {
                    // A filtered constant is not constant from time zero because the filter
                    // chain has a start-up transient, matching the other filters' routing.
                    Signal::Latency(latency) => Signal::Latency(latency * 0.5 + extra_latency),
                    Signal::Response(response, latency) => {
                        Signal::Response(response * extra_response, latency * 0.5 + extra_latency)
                    }
                    _ => Signal::Unknown,
                },
            );
        }
        output
    }

    fn ping(&mut self, probe: bool, hash: AttoHash) -> AttoHash {
        self.x.ping(probe, hash.hash(Self::ID))
    }

    fn allocate(&mut self) {
        self.x.allocate();
    }

    fn source_edge(&self, output: usize, path: Path) -> Path {
        self.x.source_edge(output, path.with_suffix(0))
    }

    fn target_edges(&self, input: usize, path: Path) -> Vec<Path> {
        self.x.target_edges(input, path.with_suffix(0))
    }

    fn fill_graph(&self, graph: &mut Graph, path: Path) {
        graph.push_node(Node::new(
            path.clone(),
            Self::ID,
            self.inputs(),
            self.outputs(),
        ));
        self.x.fill_graph(graph, path.with_suffix(0));
    }
}

// Coefficients from https://fiiir.com/, a linear phase Kaiser windowed filter with
// normalized frequency cutoff 0.22, transition band 0.06 and 80 dB stopband attenuation.
// Gain is -1.5 dB at 0.21 (18522 Hz @ 88.2 kHz) and -79 dB at 0.25.
const _HALFBAND_LINEAR_LEN: usize = 85;
#[allow(clippy::excessive_precision)]
const _HALFBAND_LINEAR: [f32; _HALFBAND_LINEAR_LEN] = [
    0.000020220200441046,
    0.000004861974285292,
    -0.000061492255405391,
    -0.000047947308542579,
    0.000111949286674030,
    0.000157238183884534,
    -0.000134029973010597,
    -0.000352195617970385,
    0.000060566750523008,
    0.000618881801452382,
    0.000195044985122935,
    -0.000886256007067166,
    -0.000711008113349721,
    0.001012498615544184,
    0.001513402890270113,
    -0.000793210142009061,
    -0.002525875150117118,
    0.000000000000000015,
    0.003528671213843895,
    0.001549877506957102,
    -0.004145791705015911,
    -0.003902794383274251,
    0.003875807190602988,
    0.006876814923619958,
    -0.002172107493248859,
    -0.009993447571056116,
    -0.001436081065240459,
    0.012454100941225989,
    0.007205685895214254,
    -0.013166435667418973,
    -0.015059319569344419,
    0.010793941877221932,
    0.024518894667885167,
    -0.003738570533651831,
    -0.034721013736356068,
    -0.010206021619584520,
    0.044523346752473138,
    0.035517283003499218,
    -0.052687647461759073,
    -0.087922447018580221,
    0.058103086744721588,
    0.312021697722786151,
    0.439999638527507508,
    0.312021697722786151,
    0.058103086744721588,
    -0.087922447018580221,
    -0.052687647461759073,
    0.035517283003499218,
    0.044523346752473138,
    -0.010206021619584520,
    -0.034721013736356068,
    -0.003738570533651831,
    0.024518894667885167,
    0.010793941877221932,
    -0.015059319569344403,
    -0.013166435667418973,
    0.007205685895214254,
    0.012454100941225989,
    -0.001436081065240459,
    -0.009993447571056116,
    -0.002172107493248859,
    0.006876814923619958,
    0.003875807190602985,
    -0.003902794383274251,
    -0.004145791705015916,
    0.001549877506957102,
    0.003528671213843895,
    0.000000000000000015,
    -0.002525875150117118,
    -0.000793210142009061,
    0.001513402890270113,
    0.001012498615544184,
    -0.000711008113349721,
    -0.000886256007067166,
    0.000195044985122935,
    0.000618881801452382,
    0.000060566750523008,
    -0.000352195617970385,
    -0.000134029973010597,
    0.000157238183884534,
    0.000111949286674031,
    -0.000047947308542579,
    -0.000061492255405391,
    0.000004861974285292,
    0.000020220200441046,
];

// Minimum phase version of the linear phase filter. Generated with scipy at:
// https://www.tutorialspoint.com/execute_scipy_online.php
// from scipy.signal import minimum_phase
// min_phase = minimum_phase(linear_phase, method='homomorphic')
const HALFBAND_MIN_LEN: usize = 43;
#[allow(clippy::excessive_precision)]
const HALFBAND_MIN: [f32; HALFBAND_MIN_LEN] = [
    4.73552339e-02,
    1.81988040e-01,
    3.49148434e-01,
    3.92748135e-01,
    2.18230867e-01,
    -5.31842843e-02,
    -1.79186566e-01,
    -7.34488007e-02,
    8.94524103e-02,
    1.00868556e-01,
    -2.08681451e-02,
    -8.82510989e-02,
    -2.07640777e-02,
    6.22587555e-02,
    4.07776255e-02,
    -3.52258090e-02,
    -4.57407870e-02,
    1.27033444e-02,
    4.14376136e-02,
    3.30799834e-03,
    -3.24608206e-02,
    -1.27856355e-02,
    2.21659033e-02,
    1.67803711e-02,
    -1.27406974e-02,
    -1.68177367e-02,
    5.35518220e-03,
    1.44761581e-02,
    -3.70651781e-04,
    -1.11140183e-02,
    -2.40622311e-03,
    7.71596027e-03,
    3.48227062e-03,
    -4.86763558e-03,
    -3.45536353e-03,
    2.79880054e-03,
    2.86736431e-03,
    -1.48746153e-03,
    -2.11827989e-03,
    7.72684113e-04,
    1.44384114e-03,
    -4.49807048e-04,
    -9.41945265e-04,
];

const HALFBAND_LEN_SIMD: usize = HALF_HALFBAND_LEN * 2;
pub const HALF_HALFBAND_LEN: usize = 3;
type SimdCoeffs = [[f32; 8]; HALF_HALFBAND_LEN];

/// Filter coefficients used for decimation, grouped at compile time for efficient f32x8 construction.
pub const DECIMATING_COEFFS: [[f32; 8]; HALFBAND_LEN_SIMD] = [
    [
        0.,
        0.,
        0.,
        0.,
        0.,
        HALFBAND_MIN[0],
        HALFBAND_MIN[1],
        HALFBAND_MIN[2],
    ],
    [
        HALFBAND_MIN[3],
        HALFBAND_MIN[4],
        HALFBAND_MIN[5],
        HALFBAND_MIN[6],
        HALFBAND_MIN[7],
        HALFBAND_MIN[8],
        HALFBAND_MIN[9],
        HALFBAND_MIN[10],
    ],
    [
        HALFBAND_MIN[11],
        HALFBAND_MIN[12],
        HALFBAND_MIN[13],
        HALFBAND_MIN[14],
        HALFBAND_MIN[15],
        HALFBAND_MIN[16],
        HALFBAND_MIN[17],
        HALFBAND_MIN[18],
    ],
    [
        HALFBAND_MIN[19],
        HALFBAND_MIN[20],
        HALFBAND_MIN[21],
        HALFBAND_MIN[22],
        HALFBAND_MIN[23],
        HALFBAND_MIN[24],
        HALFBAND_MIN[25],
        HALFBAND_MIN[26],
    ],
    [
        HALFBAND_MIN[27],
        HALFBAND_MIN[28],
        HALFBAND_MIN[29],
        HALFBAND_MIN[30],
        HALFBAND_MIN[31],
        HALFBAND_MIN[32],
        HALFBAND_MIN[33],
        HALFBAND_MIN[34],
    ],
    [
        HALFBAND_MIN[35],
        HALFBAND_MIN[36],
        HALFBAND_MIN[37],
        HALFBAND_MIN[38],
        HALFBAND_MIN[39],
        HALFBAND_MIN[40],
        HALFBAND_MIN[41],
        HALFBAND_MIN[42],
    ],
];

/// Filter coefficients used for interpolation on even (real) samples,
/// grouped at compile time for efficient f32x8 construction.
pub const INTERPOLATING_EVEN_COEFFS: SimdCoeffs = [
    [
        0.,
        0.,
        HALFBAND_MIN[0],
        HALFBAND_MIN[2],
        HALFBAND_MIN[4],
        HALFBAND_MIN[6],
        HALFBAND_MIN[8],
        HALFBAND_MIN[10],
    ],
    [
        HALFBAND_MIN[12],
        HALFBAND_MIN[14],
        HALFBAND_MIN[16],
        HALFBAND_MIN[18],
        HALFBAND_MIN[20],
        HALFBAND_MIN[22],
        HALFBAND_MIN[24],
        HALFBAND_MIN[26],
    ],
    [
        HALFBAND_MIN[28],
        HALFBAND_MIN[30],
        HALFBAND_MIN[32],
        HALFBAND_MIN[34],
        HALFBAND_MIN[36],
        HALFBAND_MIN[38],
        HALFBAND_MIN[40],
        HALFBAND_MIN[42],
    ],
];

/// Filter coefficients used for interpolation on odd (0-padded) samples,
/// grouped at compile time for efficient f32x8 construction.
pub const INTERPOLATING_ODD_COEFFS: SimdCoeffs = [
    [
        0.,
        0.,
        0.,
        HALFBAND_MIN[1],
        HALFBAND_MIN[3],
        HALFBAND_MIN[5],
        HALFBAND_MIN[7],
        HALFBAND_MIN[9],
    ],
    [
        HALFBAND_MIN[11],
        HALFBAND_MIN[13],
        HALFBAND_MIN[15],
        HALFBAND_MIN[17],
        HALFBAND_MIN[19],
        HALFBAND_MIN[21],
        HALFBAND_MIN[23],
        HALFBAND_MIN[25],
    ],
    [
        HALFBAND_MIN[27],
        HALFBAND_MIN[29],
        HALFBAND_MIN[31],
        HALFBAND_MIN[33],
        HALFBAND_MIN[35],
        HALFBAND_MIN[37],
        HALFBAND_MIN[39],
        HALFBAND_MIN[41],
    ],
];

// wrapping offset from the index of the latest sample to start filtering from.
// we add 1 more than the ringbuf length to the start sample so that the last sample is used.
const START_SAMPLE_OFFSET_HALF: usize = 129 - HALF_HALFBAND_LEN * 8;
const START_SAMPLE_OFFSET_FULL: usize = 129 - (HALFBAND_MIN_LEN / 8 + 1) * 8;
