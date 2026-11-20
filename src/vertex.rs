//! Vertex structure for `Net`.

use super::audiounit::*;
use super::buffer::*;
use super::math::*;
use super::net::*;
use super::realnet::*;
use super::sequencer::Fade;
use super::*;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

/// Sample-accurate delay line used for branch delay compensation.
/// The ring buffer is preallocated on the control thread, so the
/// audio thread never has to allocate while compensating.
#[derive(Clone, Default)]
pub(crate) struct DelayComp {
    /// Ring buffer of samples. Its length is `delay + 1`.
    buffer: Vec<f32>,
    /// Write index into the ring buffer.
    index: usize,
    /// Current delay in whole samples.
    delay: usize,
}

impl DelayComp {
    /// Create a new compensation line that is able to delay by up to `max_delay` samples.
    pub fn new(max_delay: usize) -> Self {
        let mut buffer = Vec::with_capacity(max_delay + 1);
        buffer.resize(max_delay + 1, 0.0);
        Self {
            buffer,
            index: 0,
            delay: 0,
        }
    }

    /// Current delay in whole samples.
    #[inline]
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Reset the line to silence. The configured delay is retained.
    pub fn reset(&mut self) {
        self.index = 0;
        self.buffer.fill(0.0);
    }

    /// Make sure the line can delay by at least `max_delay` samples.
    /// Existing contents are cleared. This may allocate and is meant
    /// to be called from the control thread only.
    pub fn allocate(&mut self, max_delay: usize) {
        if self.buffer.len() < max_delay + 1 {
            self.buffer.resize(max_delay + 1, 0.0);
        }
        self.reset();
    }

    /// Set the active delay in whole samples.
    ///
    /// If the line has not been preallocated for this many samples, it is
    /// grown and cleared here. Growing only happens while topology is being
    /// (re)computed on the control thread; an already-allocated line never
    /// allocates on the audio thread.
    ///
    /// The read position is reset to silence whenever the delay changes,
    /// so shrinking a line (for example, when compensation is disabled)
    /// cannot leave a stale read offset pointing into old data.
    #[inline]
    pub fn set_delay(&mut self, delay: usize) {
        if delay + 1 > self.buffer.len() {
            self.buffer.resize(delay + 1, 0.0);
        }
        if self.delay != delay {
            self.index = 0;
            self.buffer.fill(0.0);
            self.delay = delay;
        }
    }

    /// Delay one sample by the configured amount.
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        self.buffer[self.index] = x;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
        self.buffer[self.index]
    }

    /// Delay a block of `size` scalar samples.
    /// Samples past `size` are zeroed so stale data never leaks through.
    #[inline]
    pub fn process(&mut self, size: usize, input: &[f32], output: &mut [f32]) {
        debug_assert!(size <= input.len() && size <= output.len());
        let length = self.buffer.len();
        let mut index = self.index;
        for i in 0..size {
            self.buffer[index] = input[i];
            index += 1;
            if index >= length {
                index = 0;
            }
            output[i] = self.buffer[index];
        }
        self.index = index;
        for x in output[size..].iter_mut() {
            *x = 0.0;
        }
    }
}

#[derive(Clone)]
/// Individual AudioUnits are vertices in the graph.
pub(crate) struct Vertex {
    /// The unit.
    pub unit: Box<dyn AudioUnit>,
    /// Edges connecting into this vertex. The length is equal to the number of inputs.
    pub source: Vec<NetEdge>,
    /// Input buffers. The number of channels is equal to the number of inputs.
    pub input: BufferVec,
    /// Output buffers. The number of channels is equal to the number of outputs.
    pub output: BufferVec,
    /// Temporary output buffers. The number of channels is equal to the number of outputs.
    pub output_tmp: BufferVec,
    /// Input for tick iteration. The length is equal to the number of inputs.
    pub tick_input: Vec<f32>,
    /// Output for tick iteration. The length is equal to the number of outputs.
    pub tick_output: Vec<f32>,
    /// Another, temporary tick output. The length is equal to the number of outputs.
    pub tick_output_tmp: Vec<f32>,
    /// Stable, globally unique ID for this vertex.
    pub id: NodeId,
    /// Current phase of fading into next node in 0...1.
    pub fade_phase: f32,
    /// Node we are fading into, if any. Not applicable to frontends.
    pub next: NodeEdit,
    /// The next node we will be fading into, if any. Not applicable to frontends.
    pub latest: NodeEdit,
    /// This is set if all vertex inputs are sourced from successive outputs of the indicated node.
    /// We can then omit copying and use the source node outputs directly.
    pub source_vertex: Option<(NodeIndex, usize)>,
    /// Branch delay compensation line for each input channel.
    /// A line delays the incoming signal so that all inputs of the vertex
    /// arrive sample-aligned at the mix point. Allocated on the control thread.
    pub input_delay: Vec<DelayComp>,
    /// Whether any input channel currently needs compensation.
    /// When this is set the `source_vertex` zero-copy shortcut is not valid.
    pub compensated: bool,
    /// Network revision in which this vertex was changed last.
    pub changed: u64,
    /// Used during order determination: number of unaccounted for outputs.
    pub unplugged: usize,
    /// Used during order determination: has this vertex been ordered yet.
    pub ordered: bool,
}

impl Vertex {
    pub fn new(id: NodeId, index: NodeIndex, unit: Box<dyn AudioUnit>) -> Self {
        let inputs = unit.inputs();
        let outputs = unit.outputs();
        let mut vertex = Self {
            unit,
            source: Vec::new(),
            input: BufferVec::new(inputs),
            output: BufferVec::new(outputs),
            output_tmp: BufferVec::new(outputs),
            tick_input: vec![0.0; inputs],
            tick_output: vec![0.0; outputs],
            tick_output_tmp: vec![0.0; outputs],
            id,
            fade_phase: 0.0,
            next: NodeEdit::default(),
            latest: NodeEdit::default(),
            source_vertex: None,
            input_delay: (0..inputs).map(|_| DelayComp::new(0)).collect(),
            compensated: false,
            changed: 0,
            unplugged: 0,
            ordered: false,
        };
        for i in 0..vertex.inputs() {
            vertex.source.push(edge(Port::Zero, Port::Local(index, i)));
        }
        vertex
    }

    /// Number of input channels.
    #[inline]
    pub fn inputs(&self) -> usize {
        self.tick_input.len()
    }

    /// Number of output channels.
    #[inline]
    pub fn outputs(&self) -> usize {
        self.tick_output.len()
    }

    /// Preallocate everything.
    pub fn allocate(&mut self) {
        self.unit.allocate();
    }

    /// Calculate source vertex and source port.
    pub fn update_source_vertex(&mut self) {
        self.source_vertex = None;
        if self.inputs() == 0 {
            return;
        }
        // Delay compensation changes the timing of inputs, so the zero-copy
        // shortcut may only be used when every input is uncompensated.
        if self.compensated {
            return;
        }
        let mut source_node = 0;
        let mut source_port = 0;
        for i in 0..self.inputs() {
            let source = self.source[i].source;
            match source {
                Port::Local(node, port) => {
                    if i == 0 {
                        source_node = node;
                        source_port = port;
                    } else if source_node != node || source_port + i != port {
                        return;
                    }
                }
                _ => {
                    return;
                }
            }
        }
        self.source_vertex = Some((source_node, source_port));
    }

    /// We have faded to the next unit, now start fading to the latest unit, if any.
    #[allow(clippy::needless_if)]
    fn next_phase(&mut self, sender: &Option<Arc<Queue<NetReturn>>>) {
        let mut next = self.next.unit.take().unwrap();
        core::mem::swap(&mut self.unit, &mut next);
        if let Some(sender) = sender {
            if sender.enqueue(NetReturn::Unit(next)).is_ok() {}
        }
        self.next.fade = self.latest.fade.clone();
        self.fade_phase = 0.0;
        self.next.fade_time = self.latest.fade_time;
        core::mem::swap(&mut self.next.unit, &mut self.latest.unit);
    }

    /// Process one sample.
    #[inline]
    pub fn tick(&mut self, sample_rate: f32, sender: &Option<Arc<Queue<NetReturn>>>) {
        self.unit.tick(&self.tick_input, &mut self.tick_output);

        if let Some(next) = self.next.unit.as_deref_mut() {
            let f = self.next.fade.at(1.0 - self.fade_phase);
            for x in self.tick_output.iter_mut() {
                *x *= f;
            }
            next.tick(&self.tick_input, &mut self.tick_output_tmp);
            let f = self.next.fade.at(self.fade_phase);
            for (x, y) in self.tick_output.iter_mut().zip(self.tick_output_tmp.iter()) {
                *x += *y * f;
            }
            self.fade_phase += 1.0 / (self.next.fade_time * sample_rate);
            if self.fade_phase >= 1.0 {
                self.next_phase(sender);
            }
        }
    }

    /// Process a block of samples.
    #[inline]
    pub fn process(
        &mut self,
        size: usize,
        input: &BufferRef,
        sample_rate: f32,
        sender: &Option<Arc<Queue<NetReturn>>>,
    ) {
        self.unit
            .process(size, input, &mut self.output.buffer_mut());
        let outputs = self.outputs();
        if let Some(next) = self.next.unit.as_deref_mut() {
            let phase_left = ((1.0 - self.fade_phase) * self.next.fade_time * sample_rate) as usize;
            let n = min(size, phase_left);
            let fade_d = 1.0 / (self.next.fade_time * sample_rate);
            for channel in 0..outputs {
                let mut fade = self.fade_phase;
                match self.next.fade {
                    Fade::Power => {
                        for x in self.output.channel_f32_mut(channel)[..n].iter_mut() {
                            *x *= sine_ease(1.0 - fade);
                            fade += fade_d;
                        }
                    }
                    Fade::Smooth => {
                        for x in self.output.channel_f32_mut(channel)[..n].iter_mut() {
                            *x *= smooth5(1.0 - fade);
                            fade += fade_d;
                        }
                    }
                }
            }
            next.process(size, input, &mut self.output_tmp.buffer_mut());
            for channel in 0..self.outputs() {
                let mut fade = self.fade_phase;
                match self.next.fade {
                    Fade::Power => {
                        for (x, y) in self.output.channel_f32_mut(channel)[..n]
                            .iter_mut()
                            .zip(self.output_tmp.channel_f32(channel)[..n].iter())
                        {
                            *x += *y * sine_ease(fade);
                            fade += fade_d;
                        }
                    }
                    Fade::Smooth => {
                        for (x, y) in self.output.channel_f32_mut(channel)[..n]
                            .iter_mut()
                            .zip(self.output_tmp.channel_f32(channel)[..n].iter())
                        {
                            *x += *y * smooth5(fade);
                            fade += fade_d;
                        }
                    }
                }
                for (x, y) in self.output.channel_f32_mut(channel)[n..size]
                    .iter_mut()
                    .zip(self.output_tmp.channel_f32(channel)[n..size].iter())
                {
                    *x = *y;
                }
            }
            self.fade_phase += n as f32 / (self.next.fade_time * sample_rate);
            if phase_left <= size {
                // We don't start fading in the latest unit until the next block.
                self.next_phase(sender);
            }
        }
    }

    /// Edit this vertex.
    pub fn enqueue(&mut self, edit: &mut NodeEdit, sender: &Option<Arc<Queue<NetReturn>>>) {
        if self.next.unit.is_some() {
            // Replace the latest unit.
            if let Some(latest) = self.latest.unit.take() {
                if let Some(sender) = sender {
                    if sender.enqueue(NetReturn::Unit(latest)).is_ok() {}
                }
            }
            core::mem::swap(&mut self.latest, edit);
        } else {
            // Set the next unit.
            core::mem::swap(&mut self.next, edit);
            self.fade_phase = 0.0;
        }
    }
}
