﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using System.IO;

using BizHawk.Client.Common;
using BizHawk.Emulation.Cores.Nintendo.Gameboy;

namespace BizHawk.Client.EmuHawk
{
	public partial class ColorChooserForm : Form
	{
		ColorChooserForm()
		{
			InitializeComponent();
		}

		private readonly Color[] colors = new Color[12];

		/// <summary>
		/// gambatte's default dmg colors
		/// </summary>
		static readonly int[] DefaultCGBColors =
		{
			0x00ffffff, 0x00aaaaaa, 0x00555555, 0x00000000,
			0x00ffffff, 0x00aaaaaa, 0x00555555, 0x00000000,
			0x00ffffff, 0x00aaaaaa, 0x00555555, 0x00000000
		};

		/// <summary>
		/// bsnes's default dmg colors with slight tweaking
		/// </summary>
		static readonly int[] DefaultDMGColors =
		{
			10798341, 8956165, 1922333, 337157,
			10798341, 8956165, 1922333, 337157,
			10798341, 8956165, 1922333, 337157
		};

		private void RefreshAllBackdrops()
		{
			panel1.BackColor = colors[0];
			panel2.BackColor = colors[1];
			panel3.BackColor = colors[2];
			panel4.BackColor = colors[3];
			panel5.BackColor = colors[4];
			panel6.BackColor = colors[5];
			panel7.BackColor = colors[6];
			panel8.BackColor = colors[7];
			panel9.BackColor = colors[8];
			panel10.BackColor = colors[9];
			panel11.BackColor = colors[10];
			panel12.BackColor = colors[11];
		}

		Color betweencolor(Color left, Color right, double pos)
		{
			int R = (int)(right.R * pos + left.R * (1.0 - pos) + 0.5);
			int G = (int)(right.G * pos + left.G * (1.0 - pos) + 0.5);
			int B = (int)(right.B * pos + left.B * (1.0 - pos) + 0.5);
			int A = (int)(right.A * pos + left.A * (1.0 - pos) + 0.5);

			return Color.FromArgb(A, R, G, B);
		}

		void interpolate_colors(int firstindex, int lastindex)
		{
			for (int i = firstindex + 1; i < lastindex; i++)
			{
				double pos = (i - firstindex) / (double)(lastindex - firstindex);
				colors[i] = betweencolor(colors[firstindex], colors[lastindex], pos);
			}
			RefreshAllBackdrops();
		}

		private void button3_Click(object sender, EventArgs e)
		{
			interpolate_colors(0, 3);
		}

		private void button4_Click(object sender, EventArgs e)
		{
			interpolate_colors(4, 7);
		}

		private void button5_Click(object sender, EventArgs e)
		{
			interpolate_colors(8, 11);
		}

		private void panel12_DoubleClick(object _sender, EventArgs e)
		{
			Panel sender = (Panel)_sender;

			int i;
			if (sender == panel1)
				i = 0;
			else if (sender == panel2)
				i = 1;
			else if (sender == panel3)
				i = 2;
			else if (sender == panel4)
				i = 3;
			else if (sender == panel5)
				i = 4;
			else if (sender == panel6)
				i = 5;
			else if (sender == panel7)
				i = 6;
			else if (sender == panel8)
				i = 7;
			else if (sender == panel9)
				i = 8;
			else if (sender == panel10)
				i = 9;
			else if (sender == panel11)
				i = 10;
			else if (sender == panel12)
				i = 11;
			else
				return; // i = -1;

			using (var dlg = new ColorDialog())
			{
				dlg.AllowFullOpen = true;
				dlg.AnyColor = true;
				dlg.Color = colors[i];

				// custom colors are ints, not Color structs?
				// and they don't work right unless the alpha bits are set to 0
				// and the rgb order is switched
				int[] customs = new int[12];
				for (int j = 0; j < customs.Length; j++)
					customs[j] = colors[j].R | colors[j].G << 8 | colors[j].B << 16;

				dlg.CustomColors = customs;
				dlg.FullOpen = true;

				var result = dlg.ShowDialog(this);

				if (result == DialogResult.OK)
				{
					if (colors[i] != dlg.Color)
					{
						colors[i] = dlg.Color;
						sender.BackColor = colors[i];
					}
				}
			}
		}

		/// <summary>
		/// ini keys for gambatte palette file
		/// </summary>
		private static readonly string[] paletteinikeys =
		{
			"Background0",
			"Background1",
			"Background2",
			"Background3",
			"Sprite%2010",
			"Sprite%2011",
			"Sprite%2012",
			"Sprite%2013",
			"Sprite%2020",
			"Sprite%2021",
			"Sprite%2022",
			"Sprite%2023"
		};

		/// <summary>
		/// load gambatte-style .pal file
		/// </summary>
		/// <param name="f"></param>
		/// <returns>null on failure</returns>
		public static int[] LoadPalFile(TextReader f)
		{
			Dictionary<string, int> lines = new Dictionary<string, int>();

			string line;
			while ((line = f.ReadLine()) != null)
			{
				int i = line.IndexOf('=');
				if (i < 0)
					continue;
				try
				{
					lines.Add(line.Substring(0, i), int.Parse(line.Substring(i + 1)));
				}
				catch (FormatException)
				{
				}
			}

			int[] ret = new int[12];
			try
			{
				for (int i = 0; i < 12; i++)
					ret[i] = lines[paletteinikeys[i]];
			}
			catch (KeyNotFoundException)
			{
				return null;
			}
			return ret;
		}

		/// <summary>
		/// save gambatte-style palette file
		/// </summary>
		/// <param name="f"></param>
		/// <param name="colors"></param>
		public static void SavePalFile(TextWriter f, int[] colors)
		{
			f.WriteLine("[General]");
			for (int i = 0; i < 12; i++)
				f.WriteLine(String.Format("{0}={1}", paletteinikeys[i], colors[i]));
		}

		void SetAllColors(int[] _colors)
		{
			// fix alpha to 255 in created color objects, else problems
			for (int i = 0; i < colors.Length; i++)
			{
				colors[i] = Color.FromArgb(255, Color.FromArgb(_colors[i]));
			}
			RefreshAllBackdrops();
		}

		static void DoColorChooserFormDialog(IWin32Window parent, Gameboy.GambatteSettings s, bool fromemu)
		{
			using (var dlg = new ColorChooserForm())
			{
				var gb = Global.Emulator as Gameboy;
				if (fromemu)
					s = gb.GetSettings();
				dlg.SetAllColors(s.GBPalette);

				var result = dlg.ShowDialog(parent);
				if (result == DialogResult.OK)
				{
					int[] colorints = new int[12];
					for (int i = 0; i < 12; i++)
						colorints[i] = dlg.colors[i].ToArgb();

					s.GBPalette = colorints;
					if (fromemu)
						gb.PutSettings(s);
				}
			}
		}

		public static void DoColorChooserFormDialog(IWin32Window parent)
		{
			DoColorChooserFormDialog(parent, null, true);
		}

		public static void DoColorChooserFormDialog(IWin32Window parent, Gameboy.GambatteSettings s)
		{
			DoColorChooserFormDialog(parent, s, false);
		}

		void LoadColorFile(string filename, bool alert)
		{
			try
			{
				using (StreamReader f = new StreamReader(filename))
				{
					int[] newcolors = LoadPalFile(f);
					if (newcolors == null)
						throw new Exception();

					SetAllColors(newcolors);
				}
			}
			catch
			{
				if (alert)
					MessageBox.Show(this, "Error loading .pal file!");
			}
		}

		void SaveColorFile(string filename)
		{
			try
			{
				using (StreamWriter f = new StreamWriter(filename))
				{
					int[] savecolors = new int[12];
					for (int i = 0; i < 12; i++)
						// clear alpha because gambatte color files don't usually contain it
						savecolors[i] = colors[i].ToArgb() & 0xffffff;
					SavePalFile(f, savecolors);
				}
			}
			catch
			{
				MessageBox.Show(this, "Error saving .pal file!");
			}
		}

		private void button6_Click(object sender, EventArgs e)
		{
			using (var ofd = new OpenFileDialog())
			{
				ofd.InitialDirectory = PathManager.MakeAbsolutePath(Global.Config.PathEntries["GB", "Palettes"].Path, "GB");
				ofd.Filter = "Gambatte Palettes (*.pal)|*.pal|All Files|*.*";
				ofd.RestoreDirectory = true;

				var result = ofd.ShowDialog(this);
				if (result == DialogResult.OK)
					LoadColorFile(ofd.FileName, true);
			}
		}

		private void ColorChooserForm_DragDrop(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.FileDrop))
			{
				string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);

				if (files.Length > 1)
					return;
				LoadColorFile(files[0], true);
			}
		}

		private void ColorChooserForm_DragEnter(object sender, DragEventArgs e)
		{
			if (e.Data.GetDataPresent(DataFormats.FileDrop))
				e.Effect = DragDropEffects.Move;
			else
				e.Effect = DragDropEffects.None;
		}

		private void button7_Click(object sender, EventArgs e)
		{
			using (var sfd = new SaveFileDialog())
			{
				sfd.InitialDirectory = PathManager.MakeAbsolutePath(Global.Config.PathEntries["GB", "Palettes"].Path, "GB");
				sfd.FileName = Global.Game.Name + ".pal";

				sfd.Filter = "Gambatte Palettes (*.pal)|*.pal|All Files|*.*";
				sfd.RestoreDirectory = true;
				var result = sfd.ShowDialog(this);
				if (result == DialogResult.OK)
					SaveColorFile(sfd.FileName);
			}
		}

		private void OK_Click(object sender, EventArgs e)
		{
		}

		private void DefaultButton_Click(object sender, EventArgs e)
		{
			SetAllColors(DefaultDMGColors);
		}

		private void DefaultButtonCGB_Click(object sender, EventArgs e)
		{
			SetAllColors(DefaultCGBColors);
		}

		private void ColorChooserForm_Load(object sender, EventArgs e)
		{

		}
	}
}
