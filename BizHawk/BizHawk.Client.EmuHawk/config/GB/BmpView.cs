﻿using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.IO;
using System.Drawing;
using System.Windows.Forms;

using BizHawk.Client.Common;
using BizHawk.Client.EmuHawk.WinFormExtensions;

namespace BizHawk.Client.EmuHawk
{
	public partial class BmpView : Control
	{
		[Browsable(false)]
		public Bitmap bmp { get; private set; }

		bool scaled;

		public BmpView()
		{
			if (Process.GetCurrentProcess().ProcessName == "devenv")
			{
				// in the designer
				//this.BackColor = Color.Black;
				SetStyle(ControlStyles.SupportsTransparentBackColor, true);
			}
			else
			{
				SetStyle(ControlStyles.AllPaintingInWmPaint, true);
				SetStyle(ControlStyles.UserPaint, true);
				SetStyle(ControlStyles.DoubleBuffer, true);
				SetStyle(ControlStyles.SupportsTransparentBackColor, true);
				SetStyle(ControlStyles.Opaque, true);
				this.BackColor = Color.Transparent;
				this.Paint += new PaintEventHandler(BmpView_Paint);
				this.SizeChanged += new EventHandler(BmpView_SizeChanged);
				ChangeBitmapSize(1, 1);
			}
		}

		void BmpView_SizeChanged(object sender, EventArgs e)
		{
			scaled = !(bmp.Width == Width && bmp.Height == Height);
		}

		void BmpView_Paint(object sender, PaintEventArgs e)
		{
			if (scaled)
			{
				e.Graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
				e.Graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
				e.Graphics.DrawImage(bmp, 0, 0, Width, Height);
			}
			else
			{
				e.Graphics.DrawImageUnscaled(bmp, 0, 0);
			}
		}

		public void ChangeBitmapSize(Size s)
		{
			ChangeBitmapSize(s.Width, s.Height);
		}

		public void ChangeBitmapSize(int w, int h)
		{
			if (bmp != null)
			{
				if (w == bmp.Width && h == bmp.Height)
					return;
				bmp.Dispose();
			}
			bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
			BmpView_SizeChanged(null, null);
			Refresh();
		}

		public void Clear()
		{
			var lockdata = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
			Win32.MemSet(lockdata.Scan0, 0xff, (uint)(lockdata.Height * lockdata.Stride));
			bmp.UnlockBits(lockdata);
			Refresh();
		}

		public void SaveFile()
		{
			string path = PathManager.MakeAbsolutePath(
						Global.Config.PathEntries[Global.Emulator.SystemId, "Screenshots"].Path,
						Global.Emulator.SystemId);

			var di = new DirectoryInfo(path);

			if (!di.Exists)
			{
				di.Create();
			}

			var sfd = new SaveFileDialog
			{
				FileName = PathManager.FilesystemSafeName(Global.Game) + "-Palettes",
				InitialDirectory = path,
				Filter = "PNG (*.png)|*.png|Bitmap (*.bmp)|*.bmp|All Files|*.*",
				RestoreDirectory = true
			};

			var result = sfd.ShowHawkDialog();
			if (result != DialogResult.OK)
			{
				return;
			}

			var file = new FileInfo(sfd.FileName);
			var b = this.bmp;

			ImageFormat i;
			string extension = file.Extension.ToUpper();

			switch (extension)
			{
				default:
				case ".PNG":
					i = ImageFormat.Png;
					break;
				case ".BMP":
					i = ImageFormat.Bmp;
					break;
			}

			b.Save(file.FullName, i);
		}
	}
}
