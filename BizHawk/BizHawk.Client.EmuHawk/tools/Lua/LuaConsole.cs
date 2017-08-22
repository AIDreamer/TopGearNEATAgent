﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;

using BizHawk.Emulation.Common;
using BizHawk.Client.Common;
using BizHawk.Client.EmuHawk.WinFormExtensions;
using BizHawk.Client.EmuHawk.ToolExtensions;

namespace BizHawk.Client.EmuHawk
{
	public partial class LuaConsole : ToolFormBase, IToolFormAutoConfig
	{
		[RequiredService]
		private IEmulator Emulator { get; set; }

		private readonly LuaFileList _luaList;
		private bool _sortReverse;
		private string _lastColumnSorted;

		private List<string> _consoleCommandHistory = new List<string>();
		private int _consoleCommandHistoryIndex = -1;

		public bool IsRebootingCore;

		public ToolDialogSettings.ColumnList Columns { get; set; }

		public class LuaConsoleSettings
		{
			public LuaConsoleSettings()
			{
				Columns = new ToolDialogSettings.ColumnList
				{
					new ToolDialogSettings.Column { Name = "Script", Visible = true, Index = 0, Width = 92 },
					new ToolDialogSettings.Column { Name = "PathName", Visible = true, Index = 0, Width = 195 }
				};
			}

			public ToolDialogSettings.ColumnList Columns { get; set; }
		}

		[ConfigPersist]
		public LuaConsoleSettings Settings { get; set; }

		public LuaConsole()
		{
			Settings = new LuaConsoleSettings();
			_sortReverse = false;
			_lastColumnSorted = string.Empty;
			_luaList = new LuaFileList
			{
				ChangedCallback = SessionChangedCallback,
				LoadCallback = ClearOutputWindow
			};

			InitializeComponent();

			Closing += (o, e) =>
			{
				if (AskSaveChanges())
				{
					SaveColumnInfo(LuaListView, Settings.Columns);
					
					GlobalWin.DisplayManager.ClearLuaSurfaces();
					LuaImp.GuiLibrary.DrawFinish();
					CloseLua();
				}
				else
				{
					e.Cancel = true;
				}
			};

			LuaListView.QueryItemText += LuaListView_QueryItemText;
			LuaListView.QueryItemBkColor += LuaListView_QueryItemBkColor;
			LuaListView.QueryItemImage += LuaListView_QueryItemImage;
			LuaListView.QueryItemIndent += LuaListView_QueryItemIndent;
			LuaListView.VirtualMode = true;

			//this is bad, in case we ever have more than one gui part running lua.. not sure how much other badness there is like that
			LuaSandbox.DefaultLogger = ConsoleLog;
		}

		public EmuLuaLibrary LuaImp { get; set; }

		public bool UpdateBefore { get { return true; } }

		private IEnumerable<LuaFile> SelectedItems
		{
			get { return LuaListView.SelectedIndices().Select(index => _luaList[index]); }
		}

		private IEnumerable<LuaFile> SelectedFiles
		{
			get { return SelectedItems.Where(x => !x.IsSeparator); }
		}

		public void NewUpdate(ToolFormUpdateType type) { }

		public void UpdateValues()
		{
			// Do nothing
		}

		public void FastUpdate()
		{
			// Do nothing
		}

		public void ConsoleLog(string message)
		{
			OutputBox.Text += message + Environment.NewLine + Environment.NewLine;
			OutputBox.SelectionStart = OutputBox.Text.Length;
			OutputBox.ScrollToCaret();
			UpdateDialog();
		}

		private void LuaConsole_Load(object sender, EventArgs e)
		{
			if (Global.Config.RecentLuaSession.AutoLoad && !Global.Config.RecentLuaSession.Empty)
			{
					LoadSessionFromRecent(Global.Config.RecentLuaSession.MostRecent);
			}
			else if (Global.Config.RecentLua.AutoLoad)
			{
				if (!Global.Config.RecentLua.Empty)
				{
					LoadLuaFromRecent(Global.Config.RecentLua.MostRecent);
				}
			}

			LoadColumnInfo(LuaListView, Settings.Columns);
		}

		public void Restart()
		{
			// Even if the lua console is self-rebooting from client.reboot_core() we still want to re-inject dependencies
			if (IsRebootingCore)
			{
				LuaImp.Restart(Emulator.ServiceProvider);
				return;
			}

			if (LuaImp != null && LuaImp.GuiLibrary != null && LuaImp.GuiLibrary.HasLuaSurface)
			{
				LuaImp.GuiLibrary.DrawFinish();
			}

			var runningScripts = _luaList.Where(f => f.Enabled).ToList();

			foreach (var file in runningScripts)
			{
				LuaImp.CallExitEvent(file.Thread);

				var functions = LuaImp.RegisteredFunctions.Where(x => x.Lua == file.Thread).ToList();

				foreach (var function in functions)
					LuaImp.RegisteredFunctions.Remove(function);

				UpdateRegisteredFunctionsDialog();

				file.Stop();
			}

			LuaImp = new EmuLuaLibrary(Emulator.ServiceProvider);
			InputBox.AutoCompleteCustomSource.AddRange(LuaImp.Docs.Select(a => a.Library + "." + a.Name).ToArray());

			foreach (var file in runningScripts)
			{
				string pathToLoad = Path.IsPathRooted(file.Path) ? file.Path : PathManager.MakeProgramRelativePath(file.Path); //JUNIPIER SQUATCHBOX COMPLEX
				try
				{
					LuaSandbox.Sandbox(file.Thread, () =>
					{
						file.Thread = LuaImp.SpawnCoroutine(pathToLoad);
						LuaSandbox.CreateSandbox(file.Thread, Path.GetDirectoryName(pathToLoad));
						file.State = LuaFile.RunState.Running;
					}, () =>
					{
						file.State = LuaFile.RunState.Disabled;
					});
				}
				catch (Exception ex)
				{
					MessageBox.Show(ex.ToString());
				}
			}

			UpdateDialog();
		}

		private readonly List<FileSystemWatcher> _watches = new List<FileSystemWatcher>();

		private void AddFileWatches()
		{
			_watches.Clear();
			foreach (var item in _luaList)
			{
				var processedPath = PathManager.TryMakeRelative(item.Path);
				string pathToLoad = Path.IsPathRooted(processedPath)
					? processedPath
					: PathManager.MakeProgramRelativePath(processedPath);

				CreateFileWatcher(pathToLoad);
			}
		}

		private void CreateFileWatcher(string path)
		{
			var watcher = new FileSystemWatcher
			{
				Path = Path.GetDirectoryName(path),
				Filter = Path.GetFileName(path),
				NotifyFilter = NotifyFilters.LastAccess | NotifyFilters.LastWrite
							 | NotifyFilters.FileName | NotifyFilters.DirectoryName,
				EnableRaisingEvents = true,
			};

			// TODO, Deleted and Renamed events
			watcher.Changed += new FileSystemEventHandler(OnChanged);

			_watches.Add(watcher);
        }

		private void OnChanged(object source, FileSystemEventArgs e)
		{
			string message = "File: " + e.FullPath + " " + e.ChangeType;
			Invoke(new MethodInvoker(delegate
			{
				RefreshScriptMenuItem_Click(null, null);
			}));
		}

		public void LoadLuaFile(string path)
		{
			var processedPath = PathManager.TryMakeRelative(path);
			string pathToLoad = Path.IsPathRooted(processedPath) ? processedPath : PathManager.MakeProgramRelativePath(processedPath); //JUNIPIER SQUATCHBOX COMPLEX

			if (LuaAlreadyInSession(processedPath) == false)
			{
				var luaFile = new LuaFile(string.Empty, processedPath);

				_luaList.Add(luaFile);
				LuaListView.ItemCount = _luaList.Count;
				Global.Config.RecentLua.Add(processedPath);

				if (!Global.Config.DisableLuaScriptsOnLoad)
				{
					try
					{
						LuaSandbox.Sandbox(null, () =>
						{
							luaFile.Thread = LuaImp.SpawnCoroutine(pathToLoad);
							LuaSandbox.CreateSandbox(luaFile.Thread, Path.GetDirectoryName(pathToLoad));
							luaFile.State = LuaFile.RunState.Running;
						}, () =>
						{
							luaFile.State = LuaFile.RunState.Disabled;
						});
					}
					catch (Exception e)
					{

						MessageBox.Show(e.ToString());
					}
				}
				else
				{
					luaFile.State = LuaFile.RunState.Disabled;
				}

				if (Global.Config.LuaReloadOnScriptFileChange)
				{
					CreateFileWatcher(processedPath);
				}

				//luaFile.Paused = false;
			}
			else
			{
				foreach (var file in _luaList.Where(file => processedPath == file.Path && file.Enabled == false && !Global.Config.DisableLuaScriptsOnLoad))
				{
					file.Toggle();
					break;
				}

				RunLuaScripts();
			}

			UpdateDialog();
		}

		public void UpdateDialog()
		{
			LuaListView.ItemCount = _luaList.Count;
			LuaListView.Refresh();
			UpdateNumberOfScripts();
			UpdateRegisteredFunctionsDialog();
		}

		public void RunLuaScripts()
		{
			foreach (var file in _luaList)
			{
				if (!file.Enabled && file.Thread == null)
				{
					try
					{
						LuaSandbox.Sandbox(null, () =>
						{
							string pathToLoad = Path.IsPathRooted(file.Path) ? file.Path : PathManager.MakeProgramRelativePath(file.Path); //JUNIPIER SQUATCHBOX COMPLEX
							file.Thread = LuaImp.SpawnCoroutine(file.Path);
							LuaSandbox.CreateSandbox(file.Thread, Path.GetDirectoryName(pathToLoad));
						}, () =>
						{
							file.State = LuaFile.RunState.Disabled;
						});
					}
					catch (Exception e)
					{
						MessageBox.Show(e.ToString());
					}
				}
				else
				{
					file.Stop();
				}
			}
		}

		private void SessionChangedCallback()
		{
			OutputMessages.Text =
				(_luaList.Changes ? "* " : string.Empty) +
				Path.GetFileName(_luaList.Filename);
		}


		private void LuaListView_QueryItemImage(int item, int subItem, out int imageIndex)
		{
			imageIndex = -1;
			if (subItem != 0) return;
			if (_luaList[item].Paused) imageIndex = 2;
			else if (_luaList[item].Enabled) imageIndex = 1;
			else imageIndex = 0;
		}

		void LuaListView_QueryItemIndent(int item, out int itemIndent)
		{
			itemIndent = 0;
		}

		private void LuaListView_QueryItemBkColor(int index, int column, ref Color color)
		{
			if (column == 0)
			{
				if (_luaList[index].IsSeparator)
				{
					color = BackColor;
				}
				else if (_luaList[index].Enabled && !_luaList[index].Paused)
				{
					color = Color.LightCyan;
				}
				else if (_luaList[index].Enabled && _luaList[index].Paused)
				{
					color = Color.LightPink;
				}
			}

			UpdateNumberOfScripts();
		}

		private void LuaListView_QueryItemText(int index, int column, out string text)
		{
			text = string.Empty;
			if (column == 0)
			{
				text = Path.GetFileNameWithoutExtension(_luaList[index].Path); // TODO: how about allow the user to name scripts?
			}
			else if (column == 1)
			{
				text = DressUpRelative(_luaList[index].Path);
			}
		}

		private string DressUpRelative(string path)
		{
			if (path.StartsWith(".\\"))
			{
				return path.Replace(".\\", string.Empty);
			}

			return path;
		}

		private void CloseLua()
		{
			if (LuaImp != null)
			{
				LuaImp.Close();
			}
		}

		private static FileInfo GetFileFromUser(string filter)
		{
			var ofd = new OpenFileDialog
				{
					InitialDirectory = PathManager.GetLuaPath(),
					Filter = filter,
					RestoreDirectory = true
				};

			if (!Directory.Exists(ofd.InitialDirectory))
			{
				Directory.CreateDirectory(ofd.InitialDirectory);
			}

			var result = ofd.ShowHawkDialog();
			return result == DialogResult.OK ? new FileInfo(ofd.FileName) : null;
		}

		private void UpdateNumberOfScripts()
		{
			var message = string.Empty;
			var total = SelectedFiles.Count();
			var active = _luaList.Count(file => file.Enabled);
			var paused = _luaList.Count(file => file.Enabled && file.Paused);

			if (total == 1)
			{
				message += total + " script (" + active + " active, " + paused + " paused)";
			}
			else if (total == 0)
			{
				message += total + " scripts";
			}
			else
			{
				message += total + " scripts (" + active + " active, " + paused + " paused)";
			}

			NumberOfScripts.Text = message;
		}

		private void LoadLuaFromRecent(string path)
		{
			LoadLuaFile(path);
		}

		private bool LuaAlreadyInSession(string path)
		{
			return _luaList.Any(t => path == t.Path);
		}

		public void WriteToOutputWindow(string message)
		{
			if (!OutputBox.IsHandleCreated || OutputBox.IsDisposed)
			{
				return;
			}

			OutputBox.Invoke(() =>
			{
				OutputBox.Text += message;
				OutputBox.SelectionStart = OutputBox.Text.Length;
				OutputBox.ScrollToCaret();
			});
		}

		public void ClearOutputWindow()
		{
			if (!OutputBox.IsHandleCreated || OutputBox.IsDisposed)
			{
				return;
			}

			OutputBox.Invoke(() =>
			{
				OutputBox.Text = string.Empty;
				OutputBox.Refresh();
			});
		}

		public bool LoadLuaSession(string path)
		{
			var result = _luaList.LoadLuaSession(path);

			RunLuaScripts();
			UpdateDialog();
			_luaList.Changes = false;

			return result;
		}

		/// <summary>
		/// resumes suspended Co-routines
		/// </summary>
		/// <param name="includeFrameWaiters">should frame waiters be waken up? only use this immediately before a frame of emulation</param>
		public void ResumeScripts(bool includeFrameWaiters)
		{
			if (!_luaList.Any())
			{
				return;
			}

			if (LuaImp.GuiLibrary.SurfaceIsNull)
			{
				LuaImp.GuiLibrary.DrawNew("emu");
			}

			foreach (var lf in _luaList.Where(l => l.Enabled && l.Thread != null && !l.Paused))
			{
				try
				{
					LuaSandbox.Sandbox(lf.Thread, () =>
					{
						var prohibit = lf.FrameWaiting && !includeFrameWaiters;
						if (!prohibit)
						{
							var result = LuaImp.ResumeScript(lf.Thread);
							if (result.Terminated)
							{
								LuaImp.CallExitEvent(lf.Thread);
								lf.Stop();
								UpdateDialog();
							}

							lf.FrameWaiting = result.WaitForFrame;
						}
					}, () =>
					{
						lf.Stop();
					});
				}
				catch (Exception ex)
				{
					MessageBox.Show(ex.ToString());
				}
			}
		}

		public void StartLuaDrawing()
		{
			if (_luaList.Any() && LuaImp.GuiLibrary.SurfaceIsNull)
			{
				LuaImp.GuiLibrary.DrawNew("emu");
			}
		}

		public void EndLuaDrawing()
		{
			if (_luaList.Any())
			{
				LuaImp.GuiLibrary.DrawFinish();
			}
		}

		public bool WaitOne(int timeout)
		{
			if (!IsHandleCreated || IsDisposed)
			{
				return true;
			}

			return LuaImp.LuaWait.WaitOne(timeout);
		}

		private FileInfo GetSaveFileFromUser()
		{
			var sfd = new SaveFileDialog();
			if (!string.IsNullOrWhiteSpace(_luaList.Filename))
			{
				sfd.FileName = Path.GetFileNameWithoutExtension(_luaList.Filename);
				sfd.InitialDirectory = Path.GetDirectoryName(_luaList.Filename);
			}
			else if (Global.Game != null)
			{
				sfd.FileName = PathManager.FilesystemSafeName(Global.Game);
				sfd.InitialDirectory = PathManager.GetLuaPath();
			}
			else
			{
				sfd.FileName = "NULL";
				sfd.InitialDirectory = PathManager.GetLuaPath();
			}

			sfd.Filter = "Lua Session Files (*.luases)|*.luases|All Files|*.*";
			sfd.RestoreDirectory = true;
			var result = sfd.ShowHawkDialog();
			if (result != DialogResult.OK)
			{
				return null;
			}

			return new FileInfo(sfd.FileName);
		}

		private void SaveSessionAs()
		{
			var file = GetSaveFileFromUser();
			if (file != null)
			{
				_luaList.SaveSession(file.FullName);
				OutputMessages.Text = Path.GetFileName(_luaList.Filename) + " saved.";
			}
		}

		public void LoadSessionFromRecent(string path)
		{
			var doload = true;
			if (_luaList.Changes)
			{
				doload = AskSaveChanges();
			}

			if (doload)
			{
				if (!_luaList.LoadLuaSession(path))
				{
					Global.Config.RecentLuaSession.HandleLoadError(path);
				}
				else
				{
					RunLuaScripts();
					UpdateDialog();
					_luaList.Changes = false;
				}
			}

			AddFileWatches();
        }

		public bool AskSaveChanges()
		{
			if (_luaList.Changes && !string.IsNullOrEmpty(_luaList.Filename))
			{
				GlobalWin.Sound.StopSound();
				var result = MessageBox.Show("Save changes to session?", "Lua Console", MessageBoxButtons.YesNoCancel, MessageBoxIcon.Question, MessageBoxDefaultButton.Button3);
				GlobalWin.Sound.StartSound();
				if (result == DialogResult.Yes)
				{
					if (!string.IsNullOrWhiteSpace(_luaList.Filename))
					{
						_luaList.SaveSession();
					}
					else
					{
						SaveSessionAs();
					}

					return true;
				}

				if (result == DialogResult.No)
				{
					_luaList.Changes = false;
					return true;
				}

				if (result == DialogResult.Cancel)
				{
					return false;
				}
			}

			return true;
		}

		private static void UpdateRegisteredFunctionsDialog()
		{
			foreach (var form in Application.OpenForms.OfType<LuaRegisteredFunctionsList>().ToList())
			{
				form.UpdateValues();
			}
		}

		#region Events

		#region File Menu

		private void FileSubMenu_DropDownOpened(object sender, EventArgs e)
		{
			SaveSessionMenuItem.Enabled = _luaList.Changes;
		}

		private void RecentSessionsSubMenu_DropDownOpened(object sender, EventArgs e)
		{
			RecentSessionsSubMenu.DropDownItems.Clear();
			RecentSessionsSubMenu.DropDownItems.AddRange(
				Global.Config.RecentLuaSession.RecentMenu(LoadSessionFromRecent, true));
		}

		private void RecentScriptsSubMenu_DropDownOpened(object sender, EventArgs e)
		{
			RecentScriptsSubMenu.DropDownItems.Clear();
			RecentScriptsSubMenu.DropDownItems.AddRange(
				Global.Config.RecentLua.RecentMenu(LoadLuaFromRecent, true));
		}

		private void NewSessionMenuItem_Click(object sender, EventArgs e)
		{
			var result = !_luaList.Changes || AskSaveChanges();

			if (result)
			{
				_luaList.Clear();
				ClearOutputWindow();
				UpdateDialog();
			}
		}

		private void OpenSessionMenuItem_Click(object sender, EventArgs e)
		{
			var file = GetFileFromUser("Lua Session Files (*.luases)|*.luases|All Files|*.*");
			if (file != null)
			{
				_luaList.LoadLuaSession(file.FullName);
				RunLuaScripts();
				UpdateDialog();
				_luaList.Changes = false;
			}
		}

		private void SaveSessionMenuItem_Click(object sender, EventArgs e)
		{
			if (_luaList.Changes)
			{
				if (!string.IsNullOrWhiteSpace(_luaList.Filename))
				{
					_luaList.SaveSession();
				}
				else
				{
					SaveSessionAs();
				}

				OutputMessages.Text = Path.GetFileName(_luaList.Filename) + " saved.";
			}
		}

		private void SaveSessionAsMenuItem_Click(object sender, EventArgs e)
		{
			SaveSessionAs();
		}

		private void ExitMenuItem_Click(object sender, EventArgs e)
		{
			Close();
		}

		#endregion

		#region Script

		private void ScriptSubMenu_DropDownOpened(object sender, EventArgs e)
		{
			ToggleScriptMenuItem.Enabled =
				PauseScriptMenuItem.Enabled =
				EditScriptMenuItem.Enabled =
				SelectedFiles.Any();

			RemoveScriptMenuItem.Enabled =
				DuplicateScriptMenuItem.Enabled =
				MoveUpMenuItem.Enabled =
				MoveDownMenuItem.Enabled =
				LuaListView.SelectedIndices().Any();

			SelectAllMenuItem.Enabled = _luaList.Any();
			StopAllScriptsMenuItem.Enabled = _luaList.Any(script => script.Enabled);
			RegisteredFunctionsMenuItem.Enabled = GlobalWin.Tools.LuaConsole.LuaImp.RegisteredFunctions.Any();
		}

		private void NewScriptMenuItem_Click(object sender, EventArgs e)
		{
			var sfd = new SaveFileDialog
			{
				InitialDirectory = !string.IsNullOrWhiteSpace(_luaList.Filename) ?
					Path.GetDirectoryName(_luaList.Filename) :
					PathManager.MakeAbsolutePath(Global.Config.PathEntries.LuaPathFragment, null),
				DefaultExt = ".lua",
				FileName = !string.IsNullOrWhiteSpace(_luaList.Filename) ?
					Path.GetFileNameWithoutExtension(_luaList.Filename) :
					Path.GetFileNameWithoutExtension(Global.Game.Name),
				OverwritePrompt = true,
				Filter = "Lua Scripts (*.lua)|*.lua|All Files (*.*)|*.*"
			};

			var result = sfd.ShowHawkDialog();
			if (result == DialogResult.OK
				&& !string.IsNullOrWhiteSpace(sfd.FileName))
			{
				string defaultTemplate = "while true do\n\temu.frameadvance();\nend";
				File.WriteAllText(sfd.FileName, defaultTemplate);
				_luaList.Add(new LuaFile(Path.GetFileNameWithoutExtension(sfd.FileName), sfd.FileName));
				UpdateDialog();
				System.Diagnostics.Process.Start(sfd.FileName);
			}
		}

		private void OpenScriptMenuItem_Click(object sender, EventArgs e)
		{
			var file = GetFileFromUser("Lua Scripts (*.lua)|*.lua|Text (*.text)|*.txt|All Files|*.*");
			if (file != null)
			{
				LoadLuaFile(file.FullName);
				UpdateDialog();
			}
		}

		private void ToggleScriptMenuItem_Click(object sender, EventArgs e)
		{
			var files = !SelectedFiles.Any() && Global.Config.ToggleAllIfNoneSelected ? _luaList : SelectedFiles;
			foreach (var item in files)
			{
				item.Toggle();

				if (item.Enabled && item.Thread == null)
				{
					try
					{
						LuaSandbox.Sandbox(null, () =>
						{
							string pathToLoad = Path.IsPathRooted(item.Path) ? item.Path : PathManager.MakeProgramRelativePath(item.Path); //JUNIPIER SQUATCHBOX COMPLEX
							item.Thread = LuaImp.SpawnCoroutine(pathToLoad);
							LuaSandbox.CreateSandbox(item.Thread, Path.GetDirectoryName(pathToLoad));
						}, () =>
						{
							item.State = LuaFile.RunState.Disabled;
						});

						// Shenanigans
						// We want any gui.text messages from a script to immediately update even when paused
						GlobalWin.OSD.ClearGUIText();
						GlobalWin.Tools.UpdateToolsAfter();
						EndLuaDrawing();
						StartLuaDrawing();

					}
					catch (IOException)
					{
						ConsoleLog("Unable to access file " + item.Path);
					}
					catch (Exception ex)
					{
						MessageBox.Show(ex.ToString());
					}
				}
				else if (!item.Enabled && item.Thread != null)
				{
					LuaImp.CallExitEvent(item.Thread);

					var items = SelectedItems.ToList();
					foreach (var sitem in items)
					{
						var temp = sitem;
						var functions = LuaImp.RegisteredFunctions.Where(x => x.Lua == temp.Thread).ToList();
						foreach (var function in functions)
						{
							LuaImp.RegisteredFunctions.Remove(function);
						}

						UpdateRegisteredFunctionsDialog();
					}

					LuaImp.CallExitEvent(item.Thread);
					item.Stop();
					if (Global.Config.RemoveRegisteredFunctionsOnToggle)
						GlobalWin.Tools.LuaConsole.LuaImp.RegisteredFunctions.ClearAll();
				}
			}

			UpdateDialog();
			UpdateNumberOfScripts();
			LuaListView.Refresh();
		}

		private void PauseScriptMenuItem_Click(object sender, EventArgs e)
		{
			SelectedFiles.ToList().ForEach(x => x.TogglePause());
			UpdateDialog();
		}

		private void EditScriptMenuItem_Click(object sender, EventArgs e)
		{
			SelectedFiles.ToList().ForEach(file =>
			{
				// adelikat; copy/pasting from code above.  We need a method or something for this, there's probably other places we need this logic
				string pathToLoad = Path.IsPathRooted(file.Path) ? file.Path : PathManager.MakeProgramRelativePath(file.Path); //JUNIPIER SQUATCHBOX COMPLEX
				System.Diagnostics.Process.Start(pathToLoad);
			});
		}

		private void RemoveScriptMenuItem_Click(object sender, EventArgs e)
		{
			var items = SelectedItems.ToList();
			if (items.Any())
			{
				foreach (var item in items)
				{
					var temp = item;
					var functions = LuaImp.RegisteredFunctions.Where(x => x.Lua == temp.Thread).ToList();
					foreach (var function in functions)
					{
						LuaImp.RegisteredFunctions.Remove(function);
					}

					_luaList.Remove(item);
				}

				UpdateRegisteredFunctionsDialog();
				UpdateDialog();
			}
		}

		private void DuplicateScriptMenuItem_Click(object sender, EventArgs e)
		{
			if (LuaListView.SelectedIndices().Any())
			{
				var script = SelectedFiles.First();

				var sfd = new SaveFileDialog
				{
					InitialDirectory = Path.GetDirectoryName(script.Path),
					DefaultExt = ".lua",
					FileName = Path.GetFileNameWithoutExtension(script.Path) + " (1)",
					OverwritePrompt = true,
					Filter = "Lua Scripts (*.lua)|*.lua|All Files (*.*)|*.*"
				};

				if (sfd.ShowDialog() == DialogResult.OK)
				{
					string text = File.ReadAllText(script.Path);
					File.WriteAllText(sfd.FileName, text);
					_luaList.Add(new LuaFile(Path.GetFileNameWithoutExtension(sfd.FileName), sfd.FileName));
					UpdateDialog();
					System.Diagnostics.Process.Start(sfd.FileName);
				}
			}
		}

		private void InsertSeparatorMenuItem_Click(object sender, EventArgs e)
		{
			var indices = LuaListView.SelectedIndices().ToList();
			if (indices.Any() && indices.Last() < _luaList.Count)
			{
				_luaList.Insert(indices.Last(), LuaFile.SeparatorInstance);
			}
			else
			{
				_luaList.Add(LuaFile.SeparatorInstance);
			}

			UpdateDialog();
		}

		private void MoveUpMenuItem_Click(object sender, EventArgs e)
		{
			var indices = LuaListView.SelectedIndices().ToList();
			if (indices.Count == 0 || indices[0] == 0)
			{
				return;
			}

			foreach (var index in indices)
			{
				var file = _luaList[index];
				_luaList.Remove(file);
				_luaList.Insert(index - 1, file);
			}

			var newindices = indices.Select(t => t - 1).ToList();

			LuaListView.SelectedIndices.Clear();
			foreach (var newi in newindices)
			{
				LuaListView.SelectItem(newi, true);
			}

			UpdateDialog();
		}

		private void MoveDownMenuItem_Click(object sender, EventArgs e)
		{
			var indices = LuaListView.SelectedIndices().ToList();
			if (indices.Count == 0 || indices.Last() == _luaList.Count - 1)
			{
				return;
			}

			for (var i = indices.Count - 1; i >= 0; i--)
			{
				var file = _luaList[indices[i]];
				_luaList.Remove(file);
				_luaList.Insert(indices[i] + 1, file);
			}

			var newindices = indices.Select(t => t + 1).ToList();

			LuaListView.SelectedIndices.Clear();
			foreach (var newi in newindices)
			{
				LuaListView.SelectItem(newi, true);
			}

			UpdateDialog();
		}

		private void SelectAllMenuItem_Click(object sender, EventArgs e)
		{
			LuaListView.SelectAll();
		}

		private void StopAllScriptsMenuItem_Click(object sender, EventArgs e)
		{
			_luaList.StopAllScripts();
		}

		private void RegisteredFunctionsMenuItem_Click(object sender, EventArgs e)
		{
			if (LuaImp.RegisteredFunctions.Any())
			{
				var alreadyOpen = false;
				foreach (Form form in Application.OpenForms)
				{
					if (form is LuaRegisteredFunctionsList)
					{
						alreadyOpen = true;
						form.Focus();
					}
				}

				if (!alreadyOpen)
				{
					new LuaRegisteredFunctionsList
					{
						StartLocation = this.ChildPointToScreen(LuaListView)
					}.Show();
				}
			}
		}

		#endregion

		#region Options

		private void OptionsSubMenu_DropDownOpened(object sender, EventArgs e)
		{
			DisableScriptsOnLoadMenuItem.Checked = Global.Config.DisableLuaScriptsOnLoad;
			ReturnAllIfNoneSelectedMenuItem.Checked = Global.Config.ToggleAllIfNoneSelected;
			RemoveRegisteredFunctionsOnToggleMenuItem.Checked = Global.Config.RemoveRegisteredFunctionsOnToggle;
			ReloadWhenScriptFileChangesMenuItem.Checked = Global.Config.LuaReloadOnScriptFileChange;
		}

		private void DisableScriptsOnLoadMenuItem_Click(object sender, EventArgs e)
		{
			Global.Config.DisableLuaScriptsOnLoad ^= true;
		}

		private void ToggleAllIfNoneSelectedMenuItem_Click(object sender, EventArgs e)
		{
			Global.Config.ToggleAllIfNoneSelected ^= true;
		}

		private void RemoveRegisteredFunctionsOnToggleMenuItem_Click(object sender, EventArgs e)
		{
			Global.Config.RemoveRegisteredFunctionsOnToggle ^= true;
		}

		private void ReloadWhenScriptFileChangesMenuItem_Click(object sender, EventArgs e)
		{
			Global.Config.LuaReloadOnScriptFileChange ^= true;

			if (Global.Config.LuaReloadOnScriptFileChange)
			{
				AddFileWatches();
            }
			else
			{
				_watches.Clear();
			}
		}

		private readonly LuaAutocompleteInstaller LuaAutoInstaller = new LuaAutocompleteInstaller();

		private void RegisterToTextEditorsSubMenu_DropDownOpened(object sender, EventArgs e)
		{
			// Hide until this one is implemented
			RegisterNotePadMenuItem.Visible = false;

			if (LuaAutoInstaller.IsInstalled(LuaAutocompleteInstaller.TextEditors.Sublime2))
			{
				if (LuaAutoInstaller.IsBizLuaRegistered(LuaAutocompleteInstaller.TextEditors.Sublime2))
				{
					RegisterSublimeText2MenuItem.Text = "Sublime Text 2 (installed)";
					RegisterSublimeText2MenuItem.Font = new Font(RegisterSublimeText2MenuItem.Font, FontStyle.Regular);
					RegisterSublimeText2MenuItem.Image = Properties.Resources.GreenCheck;
				}
				else
				{
					RegisterSublimeText2MenuItem.Text = "Sublime Text 2 (detected)";
					RegisterSublimeText2MenuItem.Font = new Font(RegisterSublimeText2MenuItem.Font, FontStyle.Italic);
					RegisterSublimeText2MenuItem.Image = null;
				}
			}
			else
			{
				RegisterSublimeText2MenuItem.Text = "Sublime Text 2";
				RegisterSublimeText2MenuItem.Font = new Font(RegisterSublimeText2MenuItem.Font, FontStyle.Regular);
				RegisterSublimeText2MenuItem.Image = null;
			}

			if (LuaAutoInstaller.IsInstalled(LuaAutocompleteInstaller.TextEditors.NotePad))
			{
				if (LuaAutoInstaller.IsBizLuaRegistered(LuaAutocompleteInstaller.TextEditors.NotePad))
				{
					RegisterNotePadMenuItem.Text = "Notepad++ (installed)";
					RegisterNotePadMenuItem.Font = new Font(RegisterNotePadMenuItem.Font, FontStyle.Regular);
					RegisterNotePadMenuItem.Image = Properties.Resources.GreenCheck;
				}
				else
				{
					RegisterNotePadMenuItem.Text = "Notepad++ (detected)";
					RegisterNotePadMenuItem.Font = new Font(RegisterNotePadMenuItem.Font, FontStyle.Italic);
					RegisterNotePadMenuItem.Image = null;
				}
			}
			else
			{
				RegisterNotePadMenuItem.Text = "Notepad++";
				RegisterNotePadMenuItem.Font = new Font(RegisterNotePadMenuItem.Font, FontStyle.Regular);
				RegisterNotePadMenuItem.Image = null;
			}
		}

		private void RegisterSublimeText2MenuItem_Click(object sender, EventArgs e)
		{
			LuaAutoInstaller.InstallBizLua(LuaAutocompleteInstaller.TextEditors.Sublime2);
		}

		private void RegisterNotePadMenuItem_Click(object sender, EventArgs e)
		{
			LuaAutoInstaller.InstallBizLua(LuaAutocompleteInstaller.TextEditors.NotePad);
		}

		#endregion

		#region Help

		private void FunctionsListMenuItem_Click(object sender, EventArgs e)
		{
			new LuaFunctionsForm().Show();
		}

		private void OnlineDocsMenuItem_Click(object sender, EventArgs e)
		{
			System.Diagnostics.Process.Start("http://tasvideos.org/BizHawk/LuaFunctions.html");
		}

		#endregion

		#region Toolbar and Context Menu

		private void ScriptListContextMenu_Opening(object sender, CancelEventArgs e)
		{
			ToggleScriptContextItem.Enabled =
				PauseScriptContextItem.Enabled =
				EditScriptContextItem.Enabled =
				SelectedFiles.Any();

			StopAllScriptsContextItem.Visible =
				ScriptContextSeparator.Visible =
				_luaList.Any(file => file.Enabled);
		}

		private void ConsoleContextMenu_Opening(object sender, CancelEventArgs e)
		{
			RegisteredFunctionsContextItem.Enabled = LuaImp.RegisteredFunctions.Any();
		}

		private void ClearConsoleContextItem_Click(object sender, EventArgs e)
		{
			ClearOutputWindow();
		}

		#endregion

		#region Dialog, Listview, OutputBox, InputBox

		private void LuaConsole_DragDrop(object sender, DragEventArgs e)
		{
			var filePaths = (string[])e.Data.GetData(DataFormats.FileDrop);
			try
			{
				foreach (var path in filePaths)
				{
					if (Path.GetExtension(path).ToLower() == ".lua" || Path.GetExtension(path).ToLower() == ".txt")
					{
						LoadLuaFile(path);
						UpdateDialog();
					}
					else if (Path.GetExtension(path).ToLower() == ".luases")
					{
						_luaList.LoadLuaSession(path);
						RunLuaScripts();
						UpdateDialog();
						_luaList.Changes = false;
						return;
					}
				}
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.ToString());
			}
		}

		private void LuaListView_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Delete && !e.Control && !e.Alt && !e.Shift)
			{
				RemoveScriptMenuItem_Click(null, null);
			}
			else if (e.KeyCode == Keys.A && e.Control && !e.Alt && !e.Shift) // Select All
			{
				SelectAllMenuItem_Click(null, null);
			}
			else if (e.KeyCode == Keys.F12 && !e.Control && !e.Alt && !e.Shift) // F12
			{
				RegisteredFunctionsMenuItem_Click(null, null);
			}
		}

		private void LuaListView_ItemActivate(object sender, EventArgs e)
		{
			ToggleScriptMenuItem_Click(sender, e);
		}

		private void OutputBox_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.F12 && !e.Control && !e.Alt && !e.Shift) // F12
			{
				RegisteredFunctionsMenuItem_Click(null, null);
			}
		}

		/// <summary>
		/// Sorts the column Ascending on the first click and Descending on the second click.
		/// </summary>
		private void LuaListView_ColumnClick(object sender, ColumnClickEventArgs e)
		{
			var columnToSort = LuaListView.Columns[e.Column].Text;
			var luaListTemp = new List<LuaFile>();
			if (columnToSort != _lastColumnSorted)
			{
				_sortReverse = false;
			}

			// For getting the name of the .lua file, for some reason this field is kept blank in LuaFile.cs?
			// The Name variable gets emptied again near the end just in case it would break something.
			for (var i = 0; i < _luaList.Count; i++)
			{
				var words = Regex.Split(_luaList[i].Path, ".lua");
				var split = words[0].Split(Path.DirectorySeparatorChar);

				luaListTemp.Add(_luaList[i]);
				luaListTemp[i].Name = split[split.Count() - 1];
			}

			// Script, Path
			switch (columnToSort)
			{
				case "Script":
					if (_sortReverse)
					{
						luaListTemp = luaListTemp.OrderByDescending(x => x.Name).ThenBy(x => x.Path).ToList();
					}
					else
					{
						luaListTemp = luaListTemp.OrderBy(x => x.Name).ThenBy(x => x.Path).ToList();
					}

					break;
				case "Path":
					if (_sortReverse)
					{
						luaListTemp = luaListTemp.OrderByDescending(x => x.Path).ThenBy(x => x.Name).ToList();
					}
					else
					{
						luaListTemp = luaListTemp.OrderBy(x => x.Path).ThenBy(x => x.Name).ToList();
					}

					break;
			}

			for (var i = 0; i < _luaList.Count; i++)
			{
				_luaList[i] = luaListTemp[i];
				_luaList[i].Name = string.Empty;
			}

			UpdateDialog();
			_lastColumnSorted = columnToSort;
			_sortReverse = !_sortReverse;
		}

		private void RefreshScriptMenuItem_Click(object sender, EventArgs e)
		{
			ToggleScriptMenuItem_Click(sender, e);
			ToggleScriptMenuItem_Click(sender, e);
		}

		private void InputBox_KeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Enter)
			{
				string consoleBeforeCall = OutputBox.Text;

				// TODO: Maybe make these try-catches more general
				if (!string.IsNullOrWhiteSpace(InputBox.Text))
				{
					if (InputBox.Text.Contains("emu.frameadvance("))
					{
						ConsoleLog("emu.frameadvance() can not be called from the console");
						return;
					}

					LuaSandbox.Sandbox(null, () =>
					{
						LuaImp.ExecuteString(string.Format("console.log({0})", InputBox.Text));
					}, () =>
					{
						LuaSandbox.Sandbox(null, () =>
						{
							LuaImp.ExecuteString(InputBox.Text);

							if (OutputBox.Text == consoleBeforeCall)
							{
								ConsoleLog("Command successfully executed");
							}
						});
					});

					_consoleCommandHistory.Insert(0, InputBox.Text);
					_consoleCommandHistoryIndex = -1;
					InputBox.Clear();
				}
			}
			else if (e.KeyCode == Keys.Up)
			{
				if (_consoleCommandHistoryIndex < _consoleCommandHistory.Count - 1)
				{
					_consoleCommandHistoryIndex++;
					InputBox.Text = _consoleCommandHistory[_consoleCommandHistoryIndex];
					InputBox.Select(InputBox.Text.Length, 0);
				}
				e.Handled = true;
			}
			else if (e.KeyCode == Keys.Down)
			{
				if (_consoleCommandHistoryIndex == 0)
				{
					_consoleCommandHistoryIndex--;
					InputBox.Text = "";
				}
				else if (_consoleCommandHistoryIndex > 0)
				{
					_consoleCommandHistoryIndex--;
					InputBox.Text = _consoleCommandHistory[_consoleCommandHistoryIndex];
					InputBox.Select(InputBox.Text.Length, 0);
				}
				e.Handled = true;
			}
			else if (e.KeyCode == Keys.Tab)
			{
				this.ProcessTabKey(false);
				e.Handled = true;
			}
		}

		protected override bool ProcessTabKey(bool forward)
		{
			// TODO: Make me less dirty (please)
			return false;
		}

		#endregion

		private void EraseToolbarItem_Click(object sender, EventArgs e)
		{
			GlobalWin.DisplayManager.ClearLuaSurfaces();
		}

		// Stupid designer
		protected void DragEnterWrapper(object sender, DragEventArgs e)
		{
			base.GenericDragEnter(sender, e);
		}

		#endregion
	}
}
