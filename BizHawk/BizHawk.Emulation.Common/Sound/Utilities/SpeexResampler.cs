﻿using System;
using System.Runtime.InteropServices;

namespace BizHawk.Emulation.Common
{
	/// <summary>
	/// junk wrapper around LibSpeexDSP.  quite inefficient.  will be replaced
	/// </summary>
	public class SpeexResampler : IDisposable, ISoundProvider
	{
		static class LibSpeexDSP
		{
			public const int QUALITY_MAX = 10;
			public const int QUALITY_MIN = 0;
			public const int QUALITY_DEFAULT = 4;
			public const int QUALITY_VOIP = 3;
			public const int QUALITY_DESKTOP = 5;

			public enum RESAMPLER_ERR
			{
				SUCCESS = 0,
				ALLOC_FAILED = 1,
				BAD_STATE = 2,
				INVALID_ARG = 3,
				PTR_OVERLAP = 4,
				MAX_ERROR
			};

			/// <summary>
			/// Create a new resampler with integer input and output rates.
			/// </summary>
			/// <param name="nb_channels">Number of channels to be processed</param>
			/// <param name="in_rate">Input sampling rate (integer number of Hz).</param>
			/// <param name="out_rate">Output sampling rate (integer number of Hz).</param>
			/// <param name="quality">Resampling quality between 0 and 10, where 0 has poor quality and 10 has very high quality.</param>
			/// <param name="err"></param>
			/// <returns>Newly created resampler state</returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern IntPtr speex_resampler_init(uint nb_channels, uint in_rate, uint out_rate, int quality, ref RESAMPLER_ERR err);

			/// <summary>
			/// Create a new resampler with fractional input/output rates. The sampling
			/// rate ratio is an arbitrary rational number with both the numerator and
			/// denominator being 32-bit integers.
			/// </summary>
			/// <param name="nb_channels">Number of channels to be processed</param>
			/// <param name="ratio_num">Numerator of the sampling rate ratio</param>
			/// <param name="ratio_den">Denominator of the sampling rate ratio</param>
			/// <param name="in_rate">Input sampling rate rounded to the nearest integer (in Hz).</param>
			/// <param name="out_rate">Output sampling rate rounded to the nearest integer (in Hz).</param>
			/// <param name="quality">Resampling quality between 0 and 10, where 0 has poor quality and 10 has very high quality.</param>
			/// <param name="err"></param>
			/// <returns>Newly created resampler state</returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern IntPtr speex_resampler_init_frac(uint nb_channels, uint ratio_num, uint ratio_den, uint in_rate, uint out_rate, int quality, ref RESAMPLER_ERR err);

			/// <summary>
			/// Destroy a resampler state.
			/// </summary>
			/// <param name="st">Resampler state</param>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern void speex_resampler_destroy(IntPtr st);

			/// <summary>
			/// Resample a float array. The input and output buffers must *not* overlap.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="channel_index">Index of the channel to process for the multi-channel base (0 otherwise)</param>
			/// <param name="inp">Input buffer</param>
			/// <param name="in_len">Number of input samples in the input buffer. Returns the number of samples processed</param>
			/// <param name="outp">Output buffer</param>
			/// <param name="out_len">Size of the output buffer. Returns the number of samples written</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_process_float(IntPtr st, uint channel_index, float[] inp, ref uint in_len, float[] outp, ref uint out_len);

			/// <summary>
			/// Resample an int array. The input and output buffers must *not* overlap.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="channel_index">Index of the channel to process for the multi-channel base (0 otherwise)</param>
			/// <param name="inp">Input buffer</param>
			/// <param name="in_len">Number of input samples in the input buffer. Returns the number of samples processed</param>
			/// <param name="outp">Output buffer</param>
			/// <param name="out_len">Size of the output buffer. Returns the number of samples written</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_process_int(IntPtr st, uint channel_index, short[] inp, ref uint in_len, short[] outp, ref uint out_len);

			/// <summary>
			/// Resample an interleaved float array. The input and output buffers must *not* overlap.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="inp">Input buffer</param>
			/// <param name="in_len">Number of input samples in the input buffer. Returns the number of samples processed. This is all per-channel.</param>
			/// <param name="outp">Output buffer</param>
			/// <param name="out_len">Size of the output buffer. Returns the number of samples written. This is all per-channel.</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_process_interleaved_float(IntPtr st, float[] inp, ref uint in_len, float[] outp, ref uint out_len);

			/// <summary>
			/// Resample an interleaved int array. The input and output buffers must *not* overlap.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="inp">Input buffer</param>
			/// <param name="in_len">Number of input samples in the input buffer. Returns the number of samples processed. This is all per-channel.</param>
			/// <param name="outp">Output buffer</param>
			/// <param name="out_len">Size of the output buffer. Returns the number of samples written. This is all per-channel.</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_process_interleaved_int(IntPtr st, short[] inp, ref uint in_len, short[] outp, ref uint out_len);

			/// <summary>
			/// Set (change) the input/output sampling rates (integer value).
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="in_rate">Input sampling rate (integer number of Hz).</param>
			/// <param name="out_rate">Output sampling rate (integer number of Hz).</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_set_rate(IntPtr st, uint in_rate, uint out_rate);

			/// <summary>
			/// Get the current input/output sampling rates (integer value).
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="in_rate">Input sampling rate (integer number of Hz) copied.</param>
			/// <param name="out_rate">Output sampling rate (integer number of Hz) copied.</param>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern void speex_resampler_get_rate(IntPtr st, ref uint in_rate, ref uint out_rate);

			/// <summary>
			/// Set (change) the input/output sampling rates and resampling ratio (fractional values in Hz supported).
			/// </summary>
			/// <param name="st">esampler state</param>
			/// <param name="ratio_num">Numerator of the sampling rate ratio</param>
			/// <param name="ratio_den">Denominator of the sampling rate ratio</param>
			/// <param name="in_rate">Input sampling rate rounded to the nearest integer (in Hz).</param>
			/// <param name="out_rate">Output sampling rate rounded to the nearest integer (in Hz).</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_set_rate_frac(IntPtr st, uint ratio_num, uint ratio_den, uint in_rate, uint out_rate);

			/// <summary>
			/// Get the current resampling ratio. This will be reduced to the least common denominator.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="ratio_num">Numerator of the sampling rate ratio copied</param>
			/// <param name="ratio_den">Denominator of the sampling rate ratio copied</param>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern void speex_resampler_get_ratio(IntPtr st, ref uint ratio_num, ref uint ratio_den);

			/// <summary>
			/// Set (change) the conversion quality.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="quality">Resampling quality between 0 and 10, where 0 has poor quality and 10 has very high quality.</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_set_quality(IntPtr st, int quality);

			/// <summary>
			/// Get the conversion quality.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="quality">Resampling quality between 0 and 10, where 0 has poor quality and 10 has very high quality.</param>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern void speex_resampler_get_quality(IntPtr st, ref int quality);

			/// <summary>
			/// Set (change) the input stride.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="stride">Input stride</param>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern void speex_resampler_set_input_stride(IntPtr st, uint stride);

			/// <summary>
			/// Get the input stride.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="stride">Input stride copied</param>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern void speex_resampler_get_input_stride(IntPtr st, ref uint stride);

			/// <summary>
			/// Set (change) the output stride.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="stride">Output stride</param>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern void speex_resampler_set_output_stride(IntPtr st, uint stride);

			/// <summary>
			/// Get the output stride.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <param name="stride">Output stride copied</param>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern void speex_resampler_get_output_stride(IntPtr st, ref uint stride);

			/*these two functions don't exist in our version of the dll

			/// <summary>
			/// Get the latency in input samples introduced by the resampler.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern int speex_resampler_get_input_latency(IntPtr st);

			/// <summary>
			/// Get the latency in output samples introduced by the resampler.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern int speex_resampler_get_output_latency(IntPtr st);

			*/

			/// <summary>
			/// Make sure that the first samples to go out of the resamplers don't have
			/// leading zeros. This is only useful before starting to use a newly created
			/// resampler. It is recommended to use that when resampling an audio file, as
			/// it will generate a file with the same length. For real-time processing,
			/// it is probably easier not to use this call (so that the output duration
			/// is the same for the first frame).
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_skip_zeroes(IntPtr st);

			/// <summary>
			/// Reset a resampler so a new (unrelated) stream can be processed.
			/// </summary>
			/// <param name="st">Resampler state</param>
			/// <returns></returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern RESAMPLER_ERR speex_resampler_reset_mem(IntPtr st);

			/// <summary>
			/// Returns the English meaning for an error code
			/// </summary>
			/// <param name="err">Error code</param>
			/// <returns>English string</returns>
			[DllImport("libspeexdsp.dll", CallingConvention = CallingConvention.Cdecl)]
			public static extern string speex_resampler_strerror(RESAMPLER_ERR err);
		}

		/// <summary>
		/// opaque pointer to state
		/// </summary>
		private IntPtr st = IntPtr.Zero;

		/// <summary>
		/// function to call to dispatch output
		/// </summary>
		private readonly Action<short[], int> drainer;

		// TODO: this size is roughly based on how big you can make the buffer before the snes resampling (32040.5 -> 44100) gets screwed up
		private short[] inbuf = new short[512]; //[8192]; // [512];

		private short[] outbuf;

		// for sync
		private short[] outbuf2 = new short[16];
		private int outbuf2pos = 0;

		// to accept an ISyncSoundProvder input
		private readonly ISoundProvider input;

		/// <summary>
		/// in buffer position in samples (not sample pairs)
		/// </summary>
		private int inbufpos = 0;

		/// <summary>
		/// throw an exception based on error state
		/// </summary>
		/// <param name="e"></param>
		static void CheckError(LibSpeexDSP.RESAMPLER_ERR e)
		{
			switch (e)
			{
				case LibSpeexDSP.RESAMPLER_ERR.SUCCESS:
					return;
				case LibSpeexDSP.RESAMPLER_ERR.ALLOC_FAILED:
					throw new InsufficientMemoryException("LibSpeexDSP: Alloc failed");
				case LibSpeexDSP.RESAMPLER_ERR.BAD_STATE:
					throw new Exception("LibSpeexDSP: Bad state");
				case LibSpeexDSP.RESAMPLER_ERR.INVALID_ARG:
					throw new ArgumentException("LibSpeexDSP: Bad Argument");
				case LibSpeexDSP.RESAMPLER_ERR.PTR_OVERLAP:
					throw new Exception("LibSpeexDSP: Buffers cannot overlap");
			}
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="quality">0 to 10</param>
		/// <param name="rationum">numerator of srate change ratio (inrate / outrate)</param>
		/// <param name="ratioden">demonenator of srate change ratio (inrate / outrate)</param>
		/// <param name="sratein">sampling rate in, rounded to nearest hz</param>
		/// <param name="srateout">sampling rate out, rounded to nearest hz</param>
		/// <param name="drainer">function which accepts output as produced. if null, act as an ISyncSoundProvider</param>
		/// <param name="input">source to take input from when output is requested. if null, no autofetching</param>
		public SpeexResampler(int quality, uint rationum, uint ratioden, uint sratein, uint srateout, Action<short[], int> drainer = null, ISoundProvider input = null)
		{
			if (drainer != null && input != null)
				throw new ArgumentException("Can't autofetch without being an ISyncSoundProvider?");

			LibSpeexDSP.RESAMPLER_ERR err = LibSpeexDSP.RESAMPLER_ERR.SUCCESS;
			st = LibSpeexDSP.speex_resampler_init_frac(2, rationum, ratioden, sratein, srateout, quality, ref err);

			if (st == IntPtr.Zero)
				throw new Exception("LibSpeexDSP returned null!");

			CheckError(err);

			this.drainer = drainer ?? InternalDrain;
			this.input = input;

			outbuf = new short[inbuf.Length * ratioden / rationum / 2 * 2 + 128];
		}

		/// <summary>change sampling rate on the fly</summary>
		/// <param name="rationum">numerator of srate change ratio (inrate / outrate)</param>
		/// <param name="ratioden">demonenator of srate change ratio (inrate / outrate)</param>
		/// <param name="sratein">sampling rate in, rounded to nearest hz</param>
		/// <param name="srateout">sampling rate out, rounded to nearest hz</param>
		public void ChangeRate(uint rationum, uint ratioden, uint sratein, uint srateout)
		{
			CheckError(LibSpeexDSP.speex_resampler_set_rate_frac(st, rationum, ratioden, sratein, srateout));
			outbuf = new short[inbuf.Length * ratioden / rationum / 2 * 2 + 128];
		}

		/// <summary>
		/// add a sample to the queue
		/// </summary>
		/// <param name="left"></param>
		/// <param name="right"></param>
		public void EnqueueSample(short left, short right)
		{
			inbuf[inbufpos++] = left;
			inbuf[inbufpos++] = right;

			if (inbufpos == inbuf.Length)
				Flush();
		}

		/// <summary>
		/// add multiple samples to the queue
		/// </summary>
		/// <param name="userbuf">interleaved stereo samples</param>
		/// <param name="nsamp">number of sample pairs</param>
		public void EnqueueSamples(short[] userbuf, int nsamp)
		{
			int numused = 0;
			while (numused < nsamp)
			{
				int shortstocopy = Math.Min(inbuf.Length - inbufpos, (nsamp - numused) * 2);

				Buffer.BlockCopy(userbuf, numused * 2 * sizeof(short), inbuf, inbufpos * sizeof(short), shortstocopy * sizeof(short));
				inbufpos += shortstocopy;
				numused += shortstocopy / 2;

				if (inbufpos == inbuf.Length)
					Flush();
			}
		}


		/// <summary>
		/// flush as many input samples as possible, generating output samples right now
		/// </summary>
		public void Flush()
		{
			uint inal = (uint)inbufpos / 2;

			uint outal = (uint)outbuf.Length / 2;

			LibSpeexDSP.speex_resampler_process_interleaved_int(st, inbuf, ref inal, outbuf, ref outal);

			// reset inbuf

			if (inal != inbufpos / 2)
				throw new Exception("Speexresampler didn't eat the whole array?");
			inbufpos = 0;

			//Buffer.BlockCopy(inbuf, (int)inal * 2 * sizeof(short), inbuf, 0, inbufpos - (int)inal * 2);
			//inbufpos -= (int)inal * 2;

			// dispatch outbuf
			drainer(outbuf, (int)outal);
		}

		public void Dispose()
		{
			if (st != IntPtr.Zero)
			{
				LibSpeexDSP.speex_resampler_destroy(st);
				st = IntPtr.Zero;
				GC.SuppressFinalize(this);
			}
		}

		~SpeexResampler()
		{
			Dispose();
		}

		void InternalDrain(short[] buf, int nsamp)
		{
			if (outbuf2pos + nsamp * 2 > outbuf2.Length)
			{
				short[] newbuf = new short[outbuf2pos + nsamp * 2];
				Buffer.BlockCopy(outbuf2, 0, newbuf, 0, outbuf2pos * sizeof(short));
				outbuf2 = newbuf;
			}
			Buffer.BlockCopy(buf, 0, outbuf2, outbuf2pos * sizeof(short), nsamp * 2 * sizeof(short));
			outbuf2pos += nsamp * 2;
		}

		public void GetSamplesSync(out short[] samples, out int nsamp)
		{
			if (input != null)
			{
				short[] sampin;
				int nsampin;
				input.GetSamplesSync(out sampin, out nsampin);
				EnqueueSamples(sampin, nsampin);
			}
			Flush();
			nsamp = outbuf2pos / 2;
			samples = outbuf2;
			outbuf2pos = 0;
		}

		public void DiscardSamples()
		{
			outbuf2pos = 0;
		}

		public bool CanProvideAsync
		{
			get { return false; }
		}

		public SyncSoundMode SyncMode
		{
			get { return SyncSoundMode.Sync; }
		}

		public void GetSamplesAsync(short[] samples)
		{
			throw new InvalidOperationException("Async mode is not supported.");
		}

		public void SetSyncMode(SyncSoundMode mode)
		{
			if (mode == SyncSoundMode.Async)
			{
				throw new NotSupportedException("Async mode is not supported.");
			}
		}
	}
}

