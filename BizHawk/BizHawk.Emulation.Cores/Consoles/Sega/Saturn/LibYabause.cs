﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace BizHawk.Emulation.Cores.Sega.Saturn
{
	public static class LibYabause
	{
		/// <summary>
		/// A,B,C,Start,DPad
		/// </summary>
		public enum Buttons1 : byte
		{
			B = 0x01,
			C = 0x02,
			A = 0x04,
			S = 0x08,
			U = 0x10,
			D = 0x20,
			L = 0x40,
			R = 0x80
		}

		/// <summary>
		/// X,Y,Z,Shoulders
		/// </summary>
		public enum Buttons2 : byte
		{
			L = 0x08,
			Z = 0x10,
			Y = 0x20,
			X = 0x40,
			R = 0x80
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="p11">player1</param>
		/// <param name="p12">player1</param>
		/// <param name="p21">player2</param>
		/// <param name="p22">player2</param>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_setpads(Buttons1 p11, Buttons2 p12, Buttons1 p21, Buttons2 p22);


		/// <summary>
		/// set video buffer
		/// </summary>
		/// <param name="buff">32 bit color, should persist over time.  must hold at least 704*512px in software mode, or (704*n*512*n)px
		/// in hardware mode with native factor size, or w*hpx in gl mode with explicit size</param>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_setvidbuff(IntPtr buff);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="buff">persistent location of s16 interleaved</param>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_setsndbuff(IntPtr buff);

		/// <summary>
		/// soft reset, or something like that
		/// </summary>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_softreset();

		/// <summary>
		/// hard reset, or something like that
		/// </summary>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_hardreset();


		[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
		public delegate void InputCallback();

		/// <summary>
		/// set a fcn to call every time input is read
		/// </summary>
		/// <param name="cb">execxutes right before the input read.  null to clear</param>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_setinputcallback(InputCallback cb);


		[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
		public delegate void TraceCallback(string dis, string regs);

		/// <summary>
		/// set a fcn to call every time input is read
		/// </summary>
		/// <param name="cb">execxutes right before the input read.  null to clear</param>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_settracecallback(TraceCallback cb);


		/// <summary>
		/// 
		/// </summary>
		/// <param name="fn"></param>
		/// <returns>success</returns>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern bool libyabause_loadstate(string fn);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="fn"></param>
		/// <returns>success</returns>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern bool libyabause_savestate(string fn);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="w">width of framebuffer</param>
		/// <param name="h">height of framebuffer</param>
		/// <param name="nsamp">number of sample pairs produced</param>
		/// <returns>true if lagged</returns>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern bool libyabause_frameadvance(out int w, out int h, out int nsamp);

		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_deinit();

		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern bool libyabause_savesaveram(string fn);
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern bool libyabause_loadsaveram(string fn);
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_clearsaveram();
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern bool libyabause_saveramodified();

		public struct NativeMemoryDomain
		{
			public IntPtr data;
			public string name;
			public int length;
		}

		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		static extern IntPtr libyabause_getmemoryareas();

		public static IEnumerable<NativeMemoryDomain> libyabause_getmemoryareas_ex()
		{
			var ret = new List<NativeMemoryDomain>();
			IntPtr start = libyabause_getmemoryareas();
			while (true)
			{
				var nmd = (NativeMemoryDomain)Marshal.PtrToStructure(start, typeof(NativeMemoryDomain));
				if (nmd.data == IntPtr.Zero || nmd.name == null)
					return ret.AsReadOnly();
				ret.Add(nmd);
				start += Marshal.SizeOf(typeof(NativeMemoryDomain));
			}
		}

		/// <summary>
		/// set the overall resolution. only works in gl mode and when nativefactor = 0
		/// </summary>
		/// <param name="w">width</param>
		/// <param name="h">height</param>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_glresize(int w, int h);

		/// <summary>
		/// cause the overall resolution to automatically switch to a multiple of the original console resolution, as the original console resolution changes.
		/// only applies in gl mode.
		/// </summary>
		/// <param name="n">factor, 1-4, 0 to disable.</param>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern void libyabause_glsetnativefactor(int n);


		public enum CartType : int
		{
			NONE = 0,
			DRAM8MBIT = 6,
			DRAM32MBIT = 7
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="intf">cd interface.  struct need not persist after call, but the function pointers better</param>
		/// <param name="biosfn">path to bios, pass null to use built in bios emulation</param>
		/// <param name="usegl">true for opengl</param>
		/// <param name="quickload">if true, skip bios opening</param>
		/// <param name="clocksync">if true, sync RTC to actual emulated time; if false, use real real time</param>
		/// <param name="clockbase">if non-zero, initial emulation time in unix format</param>
		/// <returns></returns>
		[DllImport("libyabause.dll", CallingConvention = CallingConvention.Cdecl)]
		public static extern bool libyabause_init(ref CDInterface intf, string biosfn, bool usegl, CartType carttype, bool quickload, bool clocksync, int clockbase);

		public struct CDInterface
		{
			public int DontTouch;
			public IntPtr DontTouch2;
			/// <summary>
			/// init cd functions
			/// </summary>
			/// <param name="unused"></param>
			/// <returns>0 on success, -1 on failure</returns>
			[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
			public delegate int Init(string unused);
			public Init InitFunc;
			/// <summary>
			/// deinit cd functions
			/// </summary>
			[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
			public delegate void DeInit();
			public DeInit DeInitFunc;
			/// <summary>
			/// 0 = cd present, spinning
			/// 1 = cd present, not spinning
			/// 2 = no cd
			/// 3 = tray open
			/// </summary>
			/// <returns></returns>
			[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
			public delegate int GetStatus();
			public GetStatus GetStatusFunc;
			/// <summary>
			/// read all TOC entries
			/// </summary>
			/// <param name="dest">place to copy to</param>
			/// <returns>number of bytes written.  should be 408</returns>
			[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
			public delegate int ReadTOC(IntPtr dest);
			public ReadTOC ReadTOCFunc;
			/// <summary>
			/// read a sector, should be 2352 bytes
			/// </summary>
			/// <param name="FAD"></param>
			/// <param name="dest"></param>
			/// <returns></returns>
			[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
			public delegate int ReadSectorFAD(int FAD, IntPtr dest);
			public ReadSectorFAD ReadSectorFADFunc;
			/// <summary>
			/// hint the next sector, for async loading
			/// </summary>
			/// <param name="FAD"></param>
			[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
			public delegate void ReadAheadFAD(int FAD);
			public ReadAheadFAD ReadAheadFADFunc;
		}
	}
}
