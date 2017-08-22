/*  Copyright 2005 Guillaume Duhamel
	Copyright 2005-2006 Theo Berkau
	Copyright 2008 Filipe Azevedo <pasnox@gmail.com>

	This file is part of Yabause.

	Yabause is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.

	Yabause is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Yabause; if not, write to the Free Software
	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
*/
#include "YabauseThread.h"
#include "Settings.h"
#include "VolatileSettings.h"
#include "ui/UIPortManager.h"
#include "ui/UIYabause.h"

#include "../peripheral.h"

#include <QStringList>
#include <QDebug>

YabauseThread::YabauseThread( QObject* o )
	: QObject( o )
{
	mPause = true;
	mTimerId = -1;
	mInit = -1;
}

YabauseThread::~YabauseThread()
{
	deInitEmulation();
}

yabauseinit_struct* YabauseThread::yabauseConf()
{
	return &mYabauseConf;
}

void YabauseThread::initEmulation()
{
	reloadSettings();
	mInit = YabauseInit( &mYabauseConf );
   SetOSDToggle(showFPS);
}

void YabauseThread::deInitEmulation()
{
	YabauseDeInit();
	mInit = -1;
}

bool YabauseThread::pauseEmulation( bool pause, bool reset )
{
	if ( mPause == pause && !reset ) {
		return true;
	}
	
	if ( mInit == 0 && reset ) {
		deInitEmulation();
	}
	
	if ( mInit < 0 ) {
		initEmulation();
	}
	
	if ( mInit < 0 )
	{
		emit error( QtYabause::translate( "Can't initialize Yabause" ), false );
		return false;
	}
	
	mPause = pause;
	
	if ( mPause ) {
		ScspMuteAudio(SCSP_MUTE_SYSTEM);
		killTimer( mTimerId );
		mTimerId = -1;
	}
	else {
		ScspUnMuteAudio(SCSP_MUTE_SYSTEM);
		mTimerId = startTimer( 0 );
	}
	
	VolatileSettings * vs = QtYabause::volatileSettings();

	if (vs->value("autostart").toBool())
	{
		if (vs->value("autostart/binary").toBool()) {
			MappedMemoryLoadExec(
				vs->value("autostart/binary/filename").toString().toLocal8Bit().constData(),
				vs->value("autostart/binary/address").toUInt());
		}
		else if (vs->value("autostart/load").toBool()) {
			YabLoadStateSlot( QtYabause::volatileSettings()->value( "General/SaveStates", getDataDirPath() ).toString().toAscii().constData(), vs->value("autostart/load/slot").toInt() );
		}
		vs->setValue("autostart", false);
	}

	emit this->pause( mPause );
	
	return true;
}

bool YabauseThread::resetEmulation()
{
	if ( mInit < 0 ) {
		return false;
	}
	
	YabauseReset();
	
	emit reset();

	return true;
}

void YabauseThread::reloadControllers()
{
	PerPortReset();
	QtYabause::clearPadsBits();
	
	Settings* settings = QtYabause::settings();
	
	for ( uint port = 1; port < 3; port++ )
	{
		settings->beginGroup( QString( "Input/Port/%1/Id" ).arg( port ) );
		QStringList ids = settings->childGroups();
		settings->endGroup();
		
		ids.sort();
		foreach ( const QString& id, ids )
		{
			uint type = settings->value( QString( UIPortManager::mSettingsType ).arg( port ).arg( id ) ).toUInt();
			
			switch ( type )
			{
				case PERPAD:
				{
					PerPad_struct* padbits = PerPadAdd( port == 1 ? &PORTDATA1 : &PORTDATA2 );
					
					settings->beginGroup( QString( "Input/Port/%1/Id/%2/Controller/%3/Key" ).arg( port ).arg( id ).arg( type ) );
					QStringList padKeys = settings->childKeys();
					settings->endGroup();
					
					padKeys.sort();
					foreach ( const QString& padKey, padKeys )
					{
						const QString key = settings->value( QString( UIPortManager::mSettingsKey ).arg( port ).arg( id ).arg( type ).arg( padKey ) ).toString();
						
						PerSetKey( key.toUInt(), padKey.toUInt(), padbits );
					}
					break;
				}
				case PERMOUSE:
					QtYabause::mainWindow()->appendLog( "Mouse controller type is not yet supported" );
					break;
				default:
					QtYabause::mainWindow()->appendLog( "Invalid controller type" );
					break;
			}
		}
	}
}

void YabauseThread::reloadSettings()
{
	//QMutexLocker l( &mMutex );
	// get settings pointer
	VolatileSettings* vs = QtYabause::volatileSettings();
	
	// reset yabause conf
	resetYabauseConf();

	// read & apply settings
	mYabauseConf.m68kcoretype = vs->value( "Advanced/M68KCore", mYabauseConf.m68kcoretype ).toInt();
	mYabauseConf.percoretype = vs->value( "Input/PerCore", mYabauseConf.percoretype ).toInt();
	mYabauseConf.sh2coretype = vs->value( "Advanced/SH2Interpreter", mYabauseConf.sh2coretype ).toInt();
	mYabauseConf.vidcoretype = vs->value( "Video/VideoCore", mYabauseConf.vidcoretype ).toInt();
	mYabauseConf.osdcoretype = vs->value( "Video/OSDCore", mYabauseConf.osdcoretype ).toInt();
	mYabauseConf.sndcoretype = vs->value( "Sound/SoundCore", mYabauseConf.sndcoretype ).toInt();
	mYabauseConf.cdcoretype = vs->value( "General/CdRom", mYabauseConf.cdcoretype ).toInt();
	mYabauseConf.carttype = vs->value( "Cartridge/Type", mYabauseConf.carttype ).toInt();
	const QString r = vs->value( "Advanced/Region", mYabauseConf.regionid ).toString();
	if ( r.isEmpty() || r == "Auto" )
		mYabauseConf.regionid = 0;
	else
	{
		switch ( r[0].toAscii() )
		{
			case 'J': mYabauseConf.regionid = 1; break;
			case 'T': mYabauseConf.regionid = 2; break;
			case 'U': mYabauseConf.regionid = 4; break;
			case 'B': mYabauseConf.regionid = 5; break;
			case 'K': mYabauseConf.regionid = 6; break;
			case 'A': mYabauseConf.regionid = 0xA; break;
			case 'E': mYabauseConf.regionid = 0xC; break;
			case 'L': mYabauseConf.regionid = 0xD; break;
		}
	}
	mYabauseConf.biospath = strdup( vs->value( "General/Bios", mYabauseConf.biospath ).toString().toAscii().constData() );
	mYabauseConf.cdpath = strdup( vs->value( "General/CdRomISO", mYabauseConf.cdpath ).toString().toAscii().constData() );   
   showFPS = vs->value( "General/ShowFPS", false ).toBool();
	mYabauseConf.buppath = strdup( vs->value( "Memory/Path", mYabauseConf.buppath ).toString().toAscii().constData() );
	mYabauseConf.mpegpath = strdup( vs->value( "MpegROM/Path", mYabauseConf.mpegpath ).toString().toAscii().constData() );
	mYabauseConf.cartpath = strdup( vs->value( "Cartridge/Path", mYabauseConf.cartpath ).toString().toAscii().constData() );
	mYabauseConf.videoformattype = vs->value( "Video/VideoFormat", mYabauseConf.videoformattype ).toInt();
	
	emit requestSize( QSize( vs->value( "Video/Width", 0 ).toInt(), vs->value( "Video/Height", 0 ).toInt() ) );
	emit requestFullscreen( vs->value( "Video/Fullscreen", false ).toBool() );

	reloadControllers();
}

bool YabauseThread::emulationRunning()
{
	//QMutexLocker l( &mMutex );
	return mTimerId != -1 && !mPause;
}

bool YabauseThread::emulationPaused()
{
	//QMutexLocker l( &mMutex );
	return mPause;
}

void YabauseThread::resetYabauseConf()
{
	// free structure
	memset( &mYabauseConf, 0, sizeof( yabauseinit_struct ) );
	// fill default structure
	mYabauseConf.m68kcoretype = M68KCORE_C68K;
	mYabauseConf.percoretype = QtYabause::defaultPERCore().id;
	mYabauseConf.sh2coretype = SH2CORE_DEFAULT;
	mYabauseConf.vidcoretype = QtYabause::defaultVIDCore().id;
	mYabauseConf.sndcoretype = QtYabause::defaultSNDCore().id;
	mYabauseConf.cdcoretype = QtYabause::defaultCDCore().id;
	mYabauseConf.carttype = CART_NONE;
	mYabauseConf.regionid = 0;
	mYabauseConf.biospath = 0;
	mYabauseConf.cdpath = 0;
	mYabauseConf.buppath = 0;
	mYabauseConf.mpegpath = 0;
	mYabauseConf.cartpath = 0;
	mYabauseConf.videoformattype = VIDEOFORMATTYPE_NTSC;
}

void YabauseThread::timerEvent( QTimerEvent* )
{
	//mPause = true;
	//mRunning = true;
	//while ( mRunning )
	{
		if ( !mPause )
			PERCore->HandleEvents();
		//else
			//msleep( 25 );
		//sleep( 0 );
	}
}
