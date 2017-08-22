﻿using System;
using Newtonsoft.Json;

using BizHawk.Common;
using BizHawk.Emulation.Common;

namespace BizHawk.Emulation.Cores.ColecoVision
{
	public partial class ColecoVision : IEmulator, IStatable, ISettable<ColecoVision.ColecoSettings, ColecoVision.ColecoSyncSettings>
	{
		public ColecoSettings GetSettings()
		{
			return Settings.Clone();
		}

		public ColecoSyncSettings GetSyncSettings()
		{
			return _syncSettings.Clone();
		}

		public bool PutSettings(ColecoSettings o)
		{
			Settings = o;
			return false;
		}

		public bool PutSyncSettings(ColecoSyncSettings o)
		{
			bool ret = o.SkipBiosIntro != _syncSettings.SkipBiosIntro;
			ret |= ColecoSyncSettings.NeedsReboot(_syncSettings, o);
			_syncSettings = o;
			return ret;
		}

		public class ColecoSettings
		{
			public ColecoSettings Clone()
			{
				return (ColecoSettings)MemberwiseClone();
			}
		}

		public ColecoSettings Settings = new ColecoSettings();
		public ColecoSyncSettings _syncSettings = new ColecoSyncSettings();

		public class ColecoSyncSettings
		{
			public bool SkipBiosIntro { get; set; }

			private string _port1 = ColecoVisionControllerDeck.DefaultControllerName;
			private string _port2 = ColecoVisionControllerDeck.DefaultControllerName;

			[JsonIgnore]
			public string Port1
			{
				get { return _port1; }
				set
				{
					if (!ColecoVisionControllerDeck.ValidControllerTypes.ContainsKey(value))
					{
						throw new InvalidOperationException("Invalid controller type: " + value);
					}

					_port1 = value;
				}
			}

			[JsonIgnore]
			public string Port2
			{
				get { return _port2; }
				set
				{
					if (!ColecoVisionControllerDeck.ValidControllerTypes.ContainsKey(value))
					{
						throw new InvalidOperationException("Invalid controller type: " + value);
					}

					_port2 = value;
				}
			}

			public ColecoSyncSettings Clone()
			{
				return (ColecoSyncSettings)MemberwiseClone();
			}

			public static bool NeedsReboot(ColecoSyncSettings x, ColecoSyncSettings y)
			{
				return !DeepEquality.DeepEquals(x, y);
			}
		}
	}
}
