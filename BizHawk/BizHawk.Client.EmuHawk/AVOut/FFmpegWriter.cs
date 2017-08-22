﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;

using BizHawk.Client.Common;
using BizHawk.Emulation.Common;

namespace BizHawk.Client.EmuHawk
{
	/// <summary>
	/// uses pipes to launch an external ffmpeg process and encode
	/// </summary>
	[VideoWriter("ffmpeg", "FFmpeg writer", "Uses an external FFMPEG process to encode video and audio.  Various formats supported.  Splits on resolution change.")]
	class FFmpegWriter : IVideoWriter
	{
		/// <summary>
		/// handle to external ffmpeg process
		/// </summary>
		Process ffmpeg;

		/// <summary>
		/// the commandline actually sent to ffmpeg; for informative purposes
		/// </summary>
		string commandline;

		/// <summary>
		/// current file segment (for multires)
		/// </summary>
		int segment;

		/// <summary>
		/// base filename before segment number is attached
		/// </summary>
		string baseName;

		/// <summary>
		/// recent lines in ffmpeg's stderr, for informative purposes
		/// </summary>
		Queue<string> stderr;

		/// <summary>
		/// number of lines of stderr to buffer
		/// </summary>
		const int consolebuffer = 5;

		/// <summary>
		/// muxer handle for the current segment
		/// </summary>
		NutMuxer muxer;

		/// <summary>
		/// codec token in use
		/// </summary>
		FFmpegWriterForm.FormatPreset token;

		/// <summary>
		/// file extension actually used
		/// </summary>
		string ext;

		public void SetFrame(int frame) { }

		public void OpenFile(string baseName)
		{
			this.baseName = System.IO.Path.Combine(
				System.IO.Path.GetDirectoryName(baseName),
				System.IO.Path.GetFileNameWithoutExtension(baseName));

			ext = System.IO.Path.GetExtension(baseName);

			segment = 0;
			OpenFileSegment();
		}
		
		/// <summary>
		/// starts an ffmpeg process and sets up associated sockets
		/// </summary>
		void OpenFileSegment()
		{
			try
			{
				ffmpeg = new Process();
#if WINDOWS
				ffmpeg.StartInfo.FileName = System.IO.Path.Combine(PathManager.GetDllDirectory(), "ffmpeg.exe");
#else
				ffmpeg.StartInfo.FileName = "ffmpeg"; // expecting native version to be in path
#endif

				string filename = String.Format("{0}_{1,4:D4}{2}", baseName, segment, ext);
				ffmpeg.StartInfo.Arguments = String.Format("-y -f nut -i - {1} \"{0}\"", filename, token.commandline);
				ffmpeg.StartInfo.CreateNoWindow = true;

				// ffmpeg sends informative display to stderr, and nothing to stdout
				ffmpeg.StartInfo.RedirectStandardError = true;
				ffmpeg.StartInfo.RedirectStandardInput = true;
				ffmpeg.StartInfo.UseShellExecute = false;

				commandline = "ffmpeg " + ffmpeg.StartInfo.Arguments;

				ffmpeg.ErrorDataReceived += new DataReceivedEventHandler(StderrHandler);

				stderr = new Queue<string>(consolebuffer);

				ffmpeg.Start();
			}
			catch
			{
				ffmpeg.Dispose();
				ffmpeg = null;
				throw;
			}
			ffmpeg.BeginErrorReadLine();

			muxer = new NutMuxer(width, height, fpsnum, fpsden, sampleRate, channels, ffmpeg.StandardInput.BaseStream);
		}

		/// <summary>
		/// saves stderr lines from ffmpeg in a short queue
		/// </summary>
		/// <param name="p"></param>
		/// <param name="line"></param>
		void StderrHandler(object p, DataReceivedEventArgs line)
		{
			if (!String.IsNullOrEmpty(line.Data))
			{
				if (stderr.Count == consolebuffer)
					stderr.Dequeue();
				stderr.Enqueue(line.Data + "\n");
			}
		}

		/// <summary>
		/// finishes an ffmpeg process
		/// </summary>
		void CloseFileSegment()
		{
			muxer.Finish();
			//ffmpeg.StandardInput.Close();

			// how long should we wait here?
			ffmpeg.WaitForExit(20000);
			ffmpeg.Dispose();
			ffmpeg = null;
			stderr = null;
			commandline = null;
			muxer = null;
		}


		public void CloseFile()
		{
			CloseFileSegment();
			baseName = null;
		}

		/// <summary>
		/// returns a string containing the commandline sent to ffmpeg and recent console (stderr) output
		/// </summary>
		/// <returns></returns>
		string ffmpeg_geterror()
		{
			if (ffmpeg.StartInfo.RedirectStandardError)
			{
				ffmpeg.CancelErrorRead();
			}
			StringBuilder s = new StringBuilder();
			s.Append(commandline);
			s.Append('\n');
			while (stderr.Count > 0)
			{
				var foo = stderr.Dequeue();
				s.Append(foo);
			}
			return s.ToString();
		}


		public void AddFrame(IVideoProvider source)
		{
			if (source.BufferWidth != width || source.BufferHeight != height)
				SetVideoParameters(source.BufferWidth, source.BufferHeight);

			if (ffmpeg.HasExited)
				throw new Exception("unexpected ffmpeg death:\n" + ffmpeg_geterror());

			var video = source.GetVideoBuffer();
			try
			{
				muxer.WriteVideoFrame(video);
			}
			catch
			{
				System.Windows.Forms.MessageBox.Show("Exception! ffmpeg history:\n" + ffmpeg_geterror());
				throw;
			}

			// have to do binary write!
			//ffmpeg.StandardInput.BaseStream.Write(b, 0, b.Length);
		}

		public IDisposable AcquireVideoCodecToken(System.Windows.Forms.IWin32Window hwnd)
		{
			return FFmpegWriterForm.DoFFmpegWriterDlg(hwnd);
		}
	
		public void SetVideoCodecToken(IDisposable token)
		{
			if (token is FFmpegWriterForm.FormatPreset)
				this.token = (FFmpegWriterForm.FormatPreset)token;
			else
				throw new ArgumentException("FFmpegWriter can only take its own codec tokens!");
		}

		/// <summary>
		/// video params
		/// </summary>
		int fpsnum, fpsden, width, height, sampleRate, channels;

		public void SetMovieParameters(int fpsnum, int fpsden)
		{
			this.fpsnum = fpsnum;
			this.fpsden = fpsden;
		}

		public void SetVideoParameters(int width, int height)
		{
			this.width = width;
			this.height = height;

			/* ffmpeg theoretically supports variable resolution videos, but in practice that's not handled very well.
			 * so we start a new segment.
			 */
			if (ffmpeg != null)
			{
				CloseFileSegment();
				segment++;
				OpenFileSegment();
			}
		}


		public void SetMetaData(string gameName, string authors, ulong lengthMS, ulong rerecords)
		{
			// can be implemented with ffmpeg "-metadata" parameter???
			// nyi
		}

		public void Dispose()
		{
			if (ffmpeg != null)
				CloseFile();
		}


		public void AddSamples(short[] samples)
		{
			if (ffmpeg.HasExited)
				throw new Exception("unexpected ffmpeg death:\n" + ffmpeg_geterror());
			if (samples.Length == 0)
			{
				// has special meaning for the muxer, so don't pass on
				return;
			}
			try
			{
				muxer.WriteAudioFrame(samples);
			}
			catch
			{
				System.Windows.Forms.MessageBox.Show("Exception! ffmpeg history:\n" + ffmpeg_geterror());
				throw;
			}
		}

		public void SetAudioParameters(int sampleRate, int channels, int bits)
		{
			if (bits != 16)
				throw new ArgumentOutOfRangeException("bits", "Sampling depth must be 16 bits!");
			this.sampleRate = sampleRate;
			this.channels = channels;
		}

		public string DesiredExtension()
		{
			// this needs to interface with the codec token
			return token.defaultext;
		}

		public void SetDefaultVideoCodecToken()
		{
			token = FFmpegWriterForm.FormatPreset.GetDefaultPreset();
		}

		public bool UsesAudio { get { return true; } }
		public bool UsesVideo { get { return true; } }
	}
}
