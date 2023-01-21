/**
 * @file fsgridhandler.cpp
 * @author James Cook, Richard Nelson
 * @brief Networking constants and globals for viewer.
 *
 * $LicenseInfo:firstyear=2006&license=viewerlgpl$
 * Based on Second Life Viewer Source Code llviewernetwork.cpp
 * Copyright (C) 2010, Linden Research, Inc.
 * With modifications Copyright (C) 2012, arminweatherwax@lavabit.com
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License only.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Linden Research, Inc., 945 Battery Street, San Francisco, CA  94111  USA
 * $/LicenseInfo$
 */


#include "llviewerprecompiledheaders.h"

#include "llcommandhandler.h"
#include "lllogininstance.h"        // to check if logged in yet
#include "llnotifications.h"
#include "llnotificationsutil.h"
#include "llviewernetwork.h"
#include "llviewercontrol.h"
#include "llsdserialize.h"
#include "llsecapi.h"
#include "llstartup.h"

#include "lltrans.h"
#include "llweb.h"
#include "llbufferstream.h"
#if LL_WINDOWS
#include <Winsock2.h>
#else
#include <unistd.h>
#endif
#include "llstartup.h"

#include "fscorehttputil.h"
#include "fspanellogin.h"
#include "lfsimfeaturehandler.h"	// <COLOSI opensim multi-currency support />
#include "llmaterialtable.h" // <FS:Beq> FIRE-31628 for access to static var

void gridDownloadError( LLSD const &aData, LLGridManager* mOwner, GridEntry* mData, LLGridManager::AddState mState )
{
    LLCore::HttpStatus status = LLCoreHttpUtil::HttpCoroutineAdapter::getStatusFromLLSD( aData );

	if (HTTP_GATEWAY_TIME_OUT == status.getType() )// gateway timeout ... well ... retry once >_>
	{
		if (LLGridManager::FETCH == mState)
		{
			mOwner->addGrid(mData,	LLGridManager::RETRY);
		}

	}
	else if (status.getType() == HTTP_NOT_MODIFIED && LLGridManager::TRYLEGACY != mState) // not modified
	{
		mOwner->addGrid(mData, LLGridManager::FINISH);
	}
	else if (HTTP_INTERNAL_ERROR ==  status.getType() && LLGridManager::LOCAL == mState) //add localhost even if its not up
	{
		mOwner->addGrid(mData, LLGridManager::FINISH);
		//since we know now that its not up we cold also start it
	}
	else if (LLGridManager::TRYLEGACY == mState) //we did TRYLEGACY and faild
	{
		LLSD args;
		args["GRID"] = mData->grid[GRID_VALUE];
		//Could not add [GRID] to the grid list.
		std::string reason_dialog = "Server didn't provide grid info: ";
		reason_dialog.append(mData->last_http_error);
		reason_dialog.append("\nPlease check if the loginuri is correct and");
		args["REASON"] = reason_dialog;
		//[REASON] contact support of [GRID].
		LLNotificationsUtil::add("CantAddGrid", args);

		LL_WARNS() << "No legacy login page. Giving up for " << mData->grid[GRID_VALUE] << LL_ENDL;
		mOwner->addGrid(mData, LLGridManager::FAIL);
	}
	else
	{
		// remember the error we got when trying to get grid info where we expect it
		std::ostringstream last_error;
		last_error << status.getType() << " " << status.getMessage();
		mData->last_http_error = last_error.str();

		mOwner->addGrid(mData, LLGridManager::TRYLEGACY);
	}
}

void gridDownloadComplete( LLSD const &aData, LLGridManager* mOwner, GridEntry* mData, LLGridManager::AddState mState )
{
	//mOwner->decResponderCount();
	LLSD header = aData[ LLCoreHttpUtil::HttpCoroutineAdapter::HTTP_RESULTS ][ LLCoreHttpUtil::HttpCoroutineAdapter::HTTP_RESULTS_HEADERS];
    LLCore::HttpStatus status = LLCoreHttpUtil::HttpCoroutineAdapter::getStatusFromLLSD( aData[ LLCoreHttpUtil::HttpCoroutineAdapter::HTTP_RESULTS ] );

    const LLSD::Binary &rawData = aData[LLCoreHttpUtil::HttpCoroutineAdapter::HTTP_RESULTS_RAW].asBinary();

	// LL_DEBUGS("GridManager") << mData->grid[GRID_VALUE] << " status: " << getStatus() << " reason: " << getReason() << LL_ENDL;
	if(LLGridManager::TRYLEGACY == mState && HTTP_OK ==  status.getType() )
	{
		mOwner->addGrid(mData, LLGridManager::SYSTEM);
	}
	else if (HTTP_OK ==  status.getType() )// OK
	{
		LL_DEBUGS("GridManager") << "Parsing gridinfo xml file from "
			<< mData->grid[GRID_VALUE] << LL_ENDL;

		std::string stringData;
		stringData.assign( rawData.begin(), rawData.end() ); // LLXMLNode::parseBuffer wants a U8*, not a const U8*, so need to copy here just to be safe
		if(LLXMLNode::parseBuffer( reinterpret_cast< U8*> ( &stringData[0] ), stringData.size(), mData->info_root, NULL))
		{
			mOwner->gridInfoResponderCB(mData);
		}
		else
		{
			LLSD args;
			args["GRID"] = mData->grid[GRID_VALUE];
			//Could not add [GRID] to the grid list.
			args["REASON"] = "Server provided broken grid info xml. Please";
			//[REASON] contact support of [GRID].
			LLNotificationsUtil::add("CantAddGrid", args);

			LL_WARNS() << " Could not parse grid info xml from server."
				<< mData->grid[GRID_VALUE] << " skipping." << LL_ENDL;
			mOwner->addGrid(mData, LLGridManager::FAIL);
		}
	}
	else if (HTTP_INTERNAL_ERROR ==  status.getType() && LLGridManager::LOCAL == mState) //add localhost even if its not up
	{
		mOwner->addGrid(mData,	LLGridManager::FINISH);
		//since we know now that its not up we cold also start it
	}
	else
	{
		gridDownloadError( aData[ LLCoreHttpUtil::HttpCoroutineAdapter::HTTP_RESULTS ], mOwner, mData, mState );
	}
}

const char* DEFAULT_LOGIN_PAGE = "https://phoenixviewer.com/app/loginV3/";

const char* SYSTEM_GRID_SLURL_BASE = "secondlife://%s/secondlife/";
const char* MAIN_GRID_SLURL_BASE = "http://maps.secondlife.com/secondlife/";
const char* SYSTEM_GRID_APP_SLURL_BASE = "secondlife:///app";

const char* DEFAULT_HOP_BASE = "hop://%s/"; // <AW: hop:// protocol>
const char* DEFAULT_SLURL_BASE = "https://%s/region/";
const char* DEFAULT_APP_SLURL_BASE = "x-grid-location-info://%s/app";

// <AW opensim>
LLGridManager::LLGridManager()
:	EGridPlatform(GP_NOTSET),
	mReadyToLogin(false),
	mCommandLineDone(false),
	mResponderCount(0)
{
	mGrid.clear() ;
}

void LLGridManager::resetGrids()
{
	initGrids();
	if (!mGrid.empty())
	{
		setGridData(mConnectedGrid);
	}
}

void LLGridManager::initGrids()
{
	mGridList = LLSD();

#ifndef SINGLEGRID
	std::string grid_file  = gDirUtilp->getExpandedFilename(LL_PATH_APP_SETTINGS,  "grids.xml");
	std::string grid_user_file = gDirUtilp->getExpandedFilename(LL_PATH_USER_SETTINGS,  "grids.user.xml");
	std::string grid_remote_file = gDirUtilp->getExpandedFilename(LL_PATH_USER_SETTINGS,  "grids.remote.xml");

	mGridFile = grid_user_file;
#endif

	initSystemGrids();
#ifndef SINGLEGRID
	initGridList(grid_file, FINISH);
	initGridList(grid_remote_file, FINISH);
	initGridList(grid_user_file, FINISH);
#endif

	if(!mCommandLineDone)
	{
		initCmdLineGrids();
	}

	// <FS:ND> FIRE-20112 in case grid was set bt settings/cmd line, set it now
	if( mStartupGrid.size() )
		setGridChoice( mStartupGrid );
	mStartupGrid = "";
	// </FS:ND>
}

void LLGridManager::initSystemGrids()
{
	addSystemGrid(LLTrans::getString("LoadingData"), "", "", "", "", DEFAULT_LOGIN_PAGE);
}

void LLGridManager::initGridList(std::string grid_file, AddState state)
{
	LLSD other_grids;
	llifstream llsd_xml;
	if (grid_file.empty())
	{
		return;
	}

	if (!gDirUtilp->fileExists(grid_file))
	{
		return;
	}

	llsd_xml.open( grid_file.c_str(), std::ios::in | std::ios::binary );

	// parse through the gridfile
	if (llsd_xml.is_open()) 
	{
		LLSDSerialize::fromXMLDocument( other_grids, llsd_xml );
		if (other_grids.isMap())
		{
			for (LLSD::map_iterator grid_itr = other_grids.beginMap(); 
				grid_itr != other_grids.endMap();
				++grid_itr)
			{
				LLSD::String key_name = grid_itr->first;
				GridEntry* grid_entry = new GridEntry;
				grid_entry->set_current = false;
				grid_entry->grid = grid_itr->second;

				LL_DEBUGS("GridManager") << "reading: " << key_name << LL_ENDL;

				try
				{
					addGrid(grid_entry, state);
					//LL_DEBUGS("GridManager") << "Added grid: " << key_name << LL_ENDL;
				}
				catch (...)
				{
				}
			}
			llsd_xml.close();
		}
	}
}

void LLGridManager::initCmdLineGrids()
{
	mCommandLineDone = true;

	// load a grid from the command line.
	// if the actual grid name is specified from the command line,
	// set it as the 'selected' grid.
	std::string grid;

#ifndef SINGLEGRID
	std::string cmd_line_grid = gSavedSettings.getString("CmdLineGridChoice");
	if (!cmd_line_grid.empty())
	{
		// try to find the grid assuming the command line parameter is
		// the case-insensitive 'label' of the grid.  ie 'Agni'
		grid = getGridByGridNick(cmd_line_grid);

		if (grid.empty())
		{
			grid = getGridByLabel(cmd_line_grid);

		}
		if (grid.empty())
		{
			// if we couldn't find it, assume the
			// requested grid is the actual grid 'name' or index,
			// which would be the dns name of the grid (for non
			// linden hosted grids)
			// If the grid isn't there, that's ok, as it will be
			// automatically added later.
			grid = cmd_line_grid;
		}
	}
	else
	{
		// if a grid was not passed in via the command line, grab it from the CurrentGrid setting.
		// if there's no current grid, that's ok as it'll be either set by the value passed
		// in via the login uri if that's specified, or will default to maingrid
		grid = gSavedSettings.getString("CurrentGrid");
		LL_DEBUGS("GridManager") << "Setting grid from last selection " << grid << LL_ENDL;
	}
#endif

	if (grid.empty())
	{
		// no grid was specified so default to maingrid
		LL_DEBUGS("GridManager") << "Setting grid to MAINGRID as no grid has been specified " << LL_ENDL;
		grid = MAINGRID;
	}
	
	// generate a 'grid list' entry for any command line parameter overrides
	// or setting overides that we'll add to the grid list or override
	// any grid list entries with.

	if (mGridList.has(grid))
	{
		LL_DEBUGS("GridManager") << "Setting commandline grid " << grid << LL_ENDL;
		setGridChoice(grid);
	}
	else
	{
		LL_DEBUGS("GridManager") << "Trying to fetch commandline grid " << grid << LL_ENDL;
		GridEntry* grid_entry = new GridEntry;
		grid_entry->set_current = true;
		grid_entry->grid = LLSD::emptyMap();	
		grid_entry->grid[GRID_VALUE] = grid;

		// add the grid with the additional values, or update the
		// existing grid if it exists with the given values
		try
		{
			addGrid(grid_entry, FETCH);
		}
		catch(LLInvalidGridName ex)
		{
		}
	}
}

LLGridManager::~LLGridManager()
{
}

void LLGridManager::getGridData(const std::string &grid, LLSD& grid_info)
{
	grid_info = mGridList[grid]; 
}

void LLGridManager::gridInfoResponderCB(GridEntry* grid_entry)
{
	for (LLXMLNode* node = grid_entry->info_root->getFirstChild(); node != NULL; node = node->getNextSibling())
	{
		std::string check;
		check = "login";
		if (node->hasName(check))
		{
			// // allow redirect but not spoofing
			// LLURI uri (node->getTextContents());
			// std::string authority = uri.authority();
			// if(! authority.empty())
			// {
			// 	grid_entry->grid[GRID_VALUE] = authority;
			// }

			grid_entry->grid[GRID_LOGIN_URI_VALUE] = LLSD::emptyArray();
			grid_entry->grid[GRID_LOGIN_URI_VALUE].append(node->getTextContents());
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_LOGIN_URI_VALUE] << LL_ENDL;
			continue;
		}
		check = "gridname";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_LABEL_VALUE] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_LABEL_VALUE] << LL_ENDL;
			continue;
		}
		check = "gridnick";
		if (node->hasName(check))
		{
			grid_entry->grid[check] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[check] << LL_ENDL;
			continue;
		}
		check = "gatekeeper";
		if (node->hasName(check))
		{
			LLURI gatekeeper(node->getTextContents());
			grid_entry->grid[check] = gatekeeper.authority();

			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[check] << LL_ENDL;
 			continue;
 		}
		check = "welcome";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_LOGIN_PAGE_VALUE] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_LOGIN_PAGE_VALUE] << LL_ENDL;
			continue;
		}
		check = GRID_REGISTER_NEW_ACCOUNT;
		if (node->hasName(check))
		{
			grid_entry->grid[check] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[check] << LL_ENDL;
			continue;
		}
		check = GRID_FORGOT_PASSWORD;
		if (node->hasName(check))
		{
			grid_entry->grid[check] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[check] << LL_ENDL;
			continue;
		}
		check = "help";
		if (node->hasName(check))
		{
			grid_entry->grid[check] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[check] << LL_ENDL;
			continue;
		}
		check = "about";
		if (node->hasName(check))
		{
			grid_entry->grid[check] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[check] << LL_ENDL;
			continue;
		}
		check = "search";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_SEARCH] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_SEARCH] << LL_ENDL;
			continue;
		}
		check = "web_profile_url";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_WEB_PROFILE_VALUE] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_WEB_PROFILE_VALUE] << LL_ENDL;
			continue;
		}
		check = "profileuri";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_WEB_PROFILE_VALUE] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_WEB_PROFILE_VALUE] << LL_ENDL;
			continue;
		}
		check = "SendGridInfoToViewerOnLogin";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_SENDGRIDINFO] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_SENDGRIDINFO] << LL_ENDL;
			continue;
		}
		check = "DirectoryFee";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_DIRECTORY_FEE] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_DIRECTORY_FEE] << LL_ENDL;
			continue;
		}
		check = "platform";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_PLATFORM] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_PLATFORM] << LL_ENDL;
			continue;
		}
		check = "message";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_MESSAGE] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_MESSAGE] << LL_ENDL;
			continue;
		}
		check = "helperuri";
		if (node->hasName(check))
		{
			grid_entry->grid[GRID_HELPER_URI_VALUE] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_HELPER_URI_VALUE] << LL_ENDL;
			//don't continue, also check the next
		}
		check = "economy";
		if (node->hasName(check))
		{	//sic! economy and helperuri is 2 names for the same
			grid_entry->grid[GRID_HELPER_URI_VALUE] = node->getTextContents();
			LL_DEBUGS("GridManager") << "[\""<<check<<"\"]: " << grid_entry->grid[GRID_HELPER_URI_VALUE] << LL_ENDL;
		}
	}

	std::string grid = grid_entry->grid[GRID_VALUE].asString();
	std::string slurl_base(llformat(DEFAULT_HOP_BASE, grid.c_str())); // <AW: hop:// protocol>
	grid_entry->grid[GRID_SLURL_BASE]= slurl_base;

	LLDate now = LLDate::now();
	grid_entry->grid["LastModified"] = now;

	addGrid(grid_entry, FINISH);
}

void LLGridManager::addGrid(const std::string& loginuri)
{
	GridEntry* grid_entry = new GridEntry;
	grid_entry->set_current = true;
	grid_entry->grid = LLSD::emptyMap();
	grid_entry->grid[GRID_VALUE] = loginuri;
	addGrid(grid_entry, FETCH);
}

/// LLGridManager::addGrid - add a grid to the grid list, populating the needed values
/// if they're not populated yet.
void LLGridManager::addGrid(GridEntry* grid_entry,  AddState state)
{
	if(!grid_entry)
	{
		LL_WARNS() << "addGrid called with NULL grid_entry. Please send a bug report." << LL_ENDL;
		state = FAIL;
	}
	if(!grid_entry->grid.has(GRID_VALUE))
	{
		state = FAIL;
	}
	else if(grid_entry->grid[GRID_VALUE].asString().empty())
	{
		state = FAIL;
	}
	else if(!grid_entry->grid.isMap())
	{
		state = FAIL;
	}
	
	if (grid_entry->grid.has("USER_DELETED") || grid_entry->grid.has("DEPRECATED"))
	{
		state = REMOVE;
	}

	if ((FETCH == state) ||(FETCHTEMP == state) || (SYSTEM == state))
	{
		std::string grid = utf8str_tolower(grid_entry->grid[GRID_VALUE]);
		// grid should be in the form of a dns address
		// but also support localhost:9000 or localhost:9000/login
		if ( !grid.empty() && grid.find_first_not_of("abcdefghijklmnopqrstuvwxyz1234567890-_.:/@% ") != std::string::npos)
		{
			// // AW: this can be helpful for debugging
			// printf("grid name: %s", grid.c_str());
			// if (grid_entry)
			// {
			// state = FAIL;
			// delete grid_entry;
			// grid_entry = NULL;
			// }
			// throw LLInvalidGridName(grid);
			LLSD args;
			args["GRID"] = grid;
			LLNotificationsUtil::add("InvalidGrid", args);
			state = FAIL;
		}

		// trim last slash
		size_t pos = grid.find_last_of("/");
		if ( (grid.length()-1) == pos )
		{
			if (!mGridList.has(grid))// deal with hand edited entries *sigh*
			{
				grid.erase(pos);
				grid_entry->grid[GRID_VALUE]  = grid;
			}
		}

		// trim region from hypergrid uris
		std::string  grid_trimmed = trimHypergrid(grid);
		if (grid_trimmed != grid)
		{
			grid = grid_trimmed;
			grid_entry->grid[GRID_VALUE]  = grid;
			grid_entry->grid["HG"] = "TRUE";
		}

		if (FETCHTEMP == state)
		{
			grid_entry->grid["FLAG_TEMPORARY"] = "TRUE";
			state = FETCH;
		}
	}

	if (FETCH == state || RETRY == state)
	{
		std::string grid = utf8str_tolower(grid_entry->grid[GRID_VALUE]);
// FIRE-24068 allow https support - based on patch by unregi resident
		std::string uri;

		std::string match = "://";
		size_t find_scheme = grid.find(match);
		if ( std::string::npos != find_scheme)
		{
			uri = grid; // assign the full URI
			grid.erase(0, find_scheme + match.length()); // trim the protocol
			grid_entry->grid[GRID_VALUE] = grid; // keep the name
		}
		else
		{
			// no protocol was specified let's assume http
			// GRID_VALUE remains unchanged
			uri = "http://" + grid;
		}

		if (std::string::npos != uri.find("lindenlab.com"))
		{
			state = SYSTEM;
		}
		else
		{
			char host_name[255];
			host_name[254] ='\0';
			gethostname(host_name, 254);
			if (std::string::npos != uri.find(host_name)||
				std::string::npos != uri.find("127.0.0.1")
				|| std::string::npos != uri.find("localhost") )
			{
				LL_DEBUGS("GridManager") << "state = LOCAL" << LL_ENDL;
				state = LOCAL;
			}
			grid_entry->grid[GRID_LOGIN_URI_VALUE] = LLSD::emptyArray();
			grid_entry->grid[GRID_LOGIN_URI_VALUE].append(uri);

			size_t pos = uri.find_last_of("/");
			if ( (uri.length()-1) != pos )
			{
				uri.append("/");
			}
			uri.append("get_grid_info");

			LL_DEBUGS("GridManager") << "get_grid_info uri: " << uri << LL_ENDL;

			time_t last_modified = 0;
			if(grid_entry->grid.has("LastModified"))
			{
				LLDate saved_value = grid_entry->grid["LastModified"];
				last_modified = (time_t)saved_value.secondsSinceEpoch();
			}
			LLCore::HttpOptions::ptr_t httpOpts(new LLCore::HttpOptions);
			httpOpts->setWantHeaders(true);
			httpOpts->setLastModified((long)last_modified);

			FSCoreHttpUtil::callbackHttpGetRaw( uri, boost::bind( gridDownloadComplete, _1, this, grid_entry, state ),
												boost::bind( gridDownloadError, _1, this, grid_entry, state ), LLCore::HttpHeaders::ptr_t(), httpOpts );
			return;
		}
	}

	if (TRYLEGACY == state)
	{
		std::string grid = utf8str_tolower(grid_entry->grid[GRID_VALUE]);
		std::string uri = "https://";
		uri.append(grid);
		size_t pos = uri.find_last_of("/");
		if ( (uri.length()-1) != pos )
		{
			uri.append("/");
		}
		uri.append("cgi-bin/login.cgi");

		LL_WARNS() << "No gridinfo found. Trying if legacy login page exists: " << uri << LL_ENDL;

		FSCoreHttpUtil::callbackHttpGetRaw( uri, boost::bind( gridDownloadComplete, _1, this, grid_entry, state ),
											boost::bind( gridDownloadError, _1, this, grid_entry, state ) );
		return;
	}

	if (FAIL != state && REMOVE != state)
	{
		std::string grid = utf8str_tolower(grid_entry->grid[GRID_VALUE]);

		// populate the other values if they don't exist
		if (!grid_entry->grid.has(GRID_LABEL_VALUE)) 
		{
			grid_entry->grid[GRID_LABEL_VALUE] = grid;
			LL_WARNS() << "No \"gridname\" found in grid info, setting to " << grid_entry->grid[GRID_LABEL_VALUE].asString() << LL_ENDL;
		}

		if (!grid_entry->grid.has(GRID_NICK_VALUE))
		{
			grid_entry->grid[GRID_NICK_VALUE] = grid;
			LL_WARNS() << "No \"gridnick\" found in grid info, setting to " << grid_entry->grid[GRID_NICK_VALUE].asString() << LL_ENDL;
		}
	}

	if (SYSTEM == state)
	{
		std::string grid = utf8str_tolower(grid_entry->grid[GRID_VALUE]);

		// if the grid data doesn't include any of the URIs, then 
		// generate them from the grid

		if (!grid_entry->grid.has(GRID_LOGIN_URI_VALUE))
		{
			grid_entry->grid[GRID_LOGIN_URI_VALUE] = LLSD::emptyArray();
			grid_entry->grid[GRID_LOGIN_URI_VALUE].append(std::string("https://") + grid + "/cgi-bin/login.cgi");
			LL_WARNS() << "Adding Legacy Login Service at:" << grid_entry->grid[GRID_LOGIN_URI_VALUE].asString() << LL_ENDL;
		}

		// Populate to the default values
		if (!grid_entry->grid.has(GRID_LOGIN_PAGE_VALUE)) 
		{
			grid_entry->grid[GRID_LOGIN_PAGE_VALUE] = std::string("http://") + grid + "/app/login/";
			LL_WARNS() << "Adding Legacy Login Screen at:" << grid_entry->grid[GRID_LOGIN_PAGE_VALUE].asString() << LL_ENDL;
		}
		if (!grid_entry->grid.has(GRID_HELPER_URI_VALUE)) 
		{
			LL_WARNS() << "Adding Legacy Economy at:" << grid_entry->grid[GRID_HELPER_URI_VALUE].asString() << LL_ENDL;
			grid_entry->grid[GRID_HELPER_URI_VALUE] = std::string("https://") + grid + "/helpers/";
		}
	}

	if (FAIL != state)
	{
		std::string grid = utf8str_tolower(grid_entry->grid[GRID_VALUE]);
	
		if (grid.empty())
		{
			state = FAIL;
		}
		else
		{
			bool list_changed = false;

			if (!grid_entry->grid.has(GRID_LOGIN_IDENTIFIER_TYPES))
			{
				// non system grids and grids that haven't already been configured with values
				// get both types of credentials.
				grid_entry->grid[GRID_LOGIN_IDENTIFIER_TYPES] = LLSD::emptyArray();
				grid_entry->grid[GRID_LOGIN_IDENTIFIER_TYPES].append(CRED_IDENTIFIER_TYPE_AGENT);
				grid_entry->grid[GRID_LOGIN_IDENTIFIER_TYPES].append(CRED_IDENTIFIER_TYPE_ACCOUNT);
			}

			bool is_current = grid_entry->set_current;
			grid_entry->set_current = false;

			if (!mGridList.has(grid)) //new grid
			{
				LL_DEBUGS("GridManager") << "new grid" << LL_ENDL;
				if (!grid_entry->grid.has("USER_DELETED")
					&& !grid_entry->grid.has("DEPRECATED"))
				{
					//finally add the grid \o/
					mGridList[grid] = grid_entry->grid;
					list_changed = true;
					LL_DEBUGS("GridManager") << "Adding new entry: " << grid << LL_ENDL;
				}
				else if (grid_entry->grid.has("DEPRECATED"))
				{
					//add the deprecated entry but hide it
					//so it doesn't get used from the user list
					mGridList[grid] = grid_entry->grid;
					LL_DEBUGS("GridManager") << "Marking entry as deprecated : " << grid << LL_ENDL;
				}
				else
				{
					LL_DEBUGS("GridManager") << "Removing entry marked for deletion: " << grid << LL_ENDL;
				}
			}
			else
			{
				LL_DEBUGS("GridManager") << "existing grid" << LL_ENDL;

				LLSD existing_grid = mGridList[grid];
				if (existing_grid.has("DEPRECATED"))
				{
					LL_DEBUGS("GridManager") << "Removing entry marked as deprecated in the fallback list: " << grid << LL_ENDL;
					mGridList[grid] = grid_entry->grid;
					list_changed = true;
				}
				else if (grid_entry->grid.has("USER_DELETED"))
				{
					// entries from the fallback list can't be deleted
					// hide them instead
					mGridList[grid] = grid_entry->grid;
					list_changed = true;
					LL_DEBUGS("GridManager") << "Entry marked for deletion: " << grid << LL_ENDL;
				}
				else if (!existing_grid.has("LastModified"))
				{
					//lack of "LastModified" means existing_grid is from fallback list,
					// assume its anyway older and override with the new entry

					mGridList[grid] = grid_entry->grid;
					list_changed = true;
					LL_DEBUGS("GridManager") << "Using custom entry: " << grid << LL_ENDL;
				}
				else if (grid_entry->grid.has("LastModified"))
				{
					LLDate testing_newer = grid_entry->grid["LastModified"];
					LLDate existing = existing_grid["LastModified"];

					LL_DEBUGS("GridManager") << "testing_newer " << testing_newer << " existing " << existing << LL_ENDL;

					if (testing_newer.secondsSinceEpoch() > existing.secondsSinceEpoch())
					{
						//existing_grid is older, override.
	
						mGridList[grid] = grid_entry->grid;
						list_changed = true;
						LL_DEBUGS("GridManager") << "Updating entry: " << grid << LL_ENDL;
					}
				}
				else
				{
					LL_DEBUGS("GridManager") << "Same or newer entry already present: " << grid << LL_ENDL;
				}
			}

			if (is_current)
			{
				LL_DEBUGS("GridManager") << "Selected grid is " << grid << LL_ENDL;
				setGridChoice(grid);
			}

			if (list_changed)
			{
				mGridListChangedSignal(true);
			}
		}
	}

// This is only of use if we want to fetch infos of entire gridlists at startup
/*
	if(grid_entry && FINISH == state || FAIL == state)
	{

		if((FINISH == state && !mCommandLineDone && 0 == mResponderCount)
			||(FAIL == state && grid_entry->set_current) )
		{
			LL_DEBUGS("GridManager") << "init CmdLineGrids"  << LL_ENDL;

			initCmdLineGrids();
		}
	}
*/
	if (FAIL == state)
	{
		mGridListChangedSignal(false);
	}

	if (grid_entry)
	{
		delete grid_entry;
		grid_entry = NULL;
	}
}

//
// LLGridManager::addSystemGrid - helper for adding a system grid.
void LLGridManager::addSystemGrid(const std::string& label,
					  const std::string& name,
					  const std::string& nick,
					  const std::string& login_uri,
					  const std::string& helper,
					  const std::string& login_page )
{
	GridEntry* grid_entry = new GridEntry;
	grid_entry->set_current = false;
	grid_entry->grid = LLSD::emptyMap();
	grid_entry->grid[GRID_VALUE] = name;
	grid_entry->grid[GRID_LABEL_VALUE] = label;
	grid_entry->grid[GRID_NICK_VALUE] = nick;
	grid_entry->grid[GRID_HELPER_URI_VALUE] = helper;
	grid_entry->grid[GRID_LOGIN_URI_VALUE] = LLSD::emptyArray();
	grid_entry->grid[GRID_LOGIN_URI_VALUE].append(login_uri);
	grid_entry->grid[GRID_LOGIN_PAGE_VALUE] = login_page;
	grid_entry->grid[GRID_IS_SYSTEM_GRID_VALUE] = true;
	grid_entry->grid[GRID_LOGIN_IDENTIFIER_TYPES] = LLSD::emptyArray();
	grid_entry->grid[GRID_LOGIN_IDENTIFIER_TYPES].append(CRED_IDENTIFIER_TYPE_AGENT);
	
	grid_entry->grid[GRID_APP_SLURL_BASE] = SYSTEM_GRID_APP_SLURL_BASE;

	
	// only add the system grids beyond agni to the visible list
	// if we're building a debug version.
	if (name == std::string(MAINGRID))
	{
		grid_entry->grid[GRID_SLURL_BASE] = MAIN_GRID_SLURL_BASE;
		grid_entry->grid[GRID_IS_FAVORITE_VALUE] = TRUE;
	}
	else
	{
		grid_entry->grid[GRID_SLURL_BASE] = llformat(SYSTEM_GRID_SLURL_BASE, label.c_str());
	}

	LL_DEBUGS("PanelLogin") << " " << grid_entry->grid[GRID_LOGIN_PAGE_VALUE]<< LL_ENDL;

	try
	{
		addGrid(grid_entry, SYSTEM);
	}
	catch(LLInvalidGridName ex)
	{
	}
}

void LLGridManager::reFetchGrid(const std::string& grid, bool set_current)
{
	GridEntry* grid_entry = new GridEntry;
	grid_entry->grid[GRID_VALUE] = grid;
	grid_entry->set_current = set_current;
	addGrid(grid_entry, FETCH);
}

void LLGridManager::removeGrid(const std::string& grid)
{
	GridEntry* grid_entry = new GridEntry;
	grid_entry->grid[GRID_VALUE] = grid;
	grid_entry->grid["USER_DELETED"]="TRUE";
	grid_entry->set_current = false;
	addGrid(grid_entry, REMOVE);
}

boost::signals2::connection LLGridManager::addGridListChangedCallback(grid_list_changed_callback_t cb)
{
	return mGridListChangedSignal.connect(cb);
}

// return a list of grid name -> grid label mappings for UI purposes
std::map<std::string, std::string> LLGridManager::getKnownGrids()
{
	std::map<std::string, std::string> result;
	for(LLSD::map_iterator grid_iter = mGridList.beginMap();
		grid_iter != mGridList.endMap();
		grid_iter++) 
	{
		if (!(grid_iter->second.has("DEPRECATED")//use in fallback list only
				||grid_iter->second.has("USER_DELETED")//use in user list only
			))
		{
			result[grid_iter->first] = grid_iter->second[GRID_LABEL_VALUE].asString();
		}
	}

	return result;
}

std::string LLGridManager::getGrid( const std::string &grid )
{
	std::string grid_name;

	if (mGridList.has(grid))
	{
		// the grid was the long name, so we're good, return it
		grid_name = grid;
	}
	else
	{
		grid_name = getGridByProbing(grid);
	}
	return grid_name;
}

std::string LLGridManager::getGridLabel(const std::string& grid)
{
	std::string grid_label;
	std::string grid_name = getGrid(grid);
	if (!grid.empty())
	{
		grid_label = mGridList[grid_name][GRID_LABEL_VALUE].asString();
	}
	else
	{
		if (LLStartUp::getStartupState() >= STATE_LOGIN_SHOW)
		{
			LL_WARNS("GridManager") << "invalid grid '" << grid << "'" << LL_ENDL;
		}
	}
	LL_DEBUGS("GridManager") << "returning " << grid_label << LL_ENDL;
	return grid_label;
}

void LLGridManager::setGridChoice(const std::string& grid)
{
	if (grid.empty() || LLLoginInstance::getInstance()->authSuccess())
	{
		return;
	}

	// <FS:ND> FIRE-20112 viewer is not properly initialized yet, store grid choice and call this method again later.
	if( mGridList.size() == 0 )
	{
		mStartupGrid = grid;
		return;
	}
	// </FS:ND>

	// Set the grid choice based on a string.
	// The string can be:
	// - a grid label from the gGridInfo table 
	// - a hostname
	// - an ip address

	// loop through.  We could do just a hash lookup but we also want to match
	// on label

	mReadyToLogin = false;
	std::string grid_name = grid;
	if (mGridList.has(grid_name))
	{
		LL_DEBUGS("GridManager") << "got grid from grid list: " << grid << LL_ENDL;
	}
	else
	{
		// case insensitive
		grid_name = getGridByGridNick(grid);
		if (!grid_name.empty())
		{
			LL_DEBUGS("GridManager") << "got grid by gridnick: " << grid << LL_ENDL;
		}
	}

	if (grid_name.empty())
	{
		// case insensitive
		grid_name = getGridByLabel(grid);
		if (!grid_name.empty())
		{
			LL_DEBUGS("GridManager") << "got grid by label: " << grid << LL_ENDL;
		}
	}
	
	if (grid_name.empty())
	{
		LL_DEBUGS("GridManager")<< "trying to fetch grid: " << grid << LL_ENDL;
		// the grid was not in the list of grids.
		GridEntry* grid_entry = new GridEntry;
		grid_entry->grid = LLSD::emptyMap();
		grid_entry->grid[GRID_VALUE] = grid;
		grid_entry->set_current = true;
		try
		{
			addGrid(grid_entry, FETCH);
		}
		catch(LLInvalidGridName ex)
		{
		}
	}
	else
	{
		LL_DEBUGS("GridManager")<< "setting grid choice: " << grid << LL_ENDL;
		mGrid = grid;// AW: don't set mGrid anywhere else
		getGridData(mConnectedGrid);
		gSavedSettings.setString("CurrentGrid", grid);
		LLTrans::setDefaultArg("CURRENT_GRID", getGridLabel()); //<FS:AW make CURRENT_GRID a default substitution>
		updateIsInProductionGrid();
		LLMaterialTable::basic.replaceCollsionSounds(isInOpenSim()); // <FS:Beq> FIRE-31628 Use OpenSim collision sounds when in OpenSim
		mReadyToLogin = true;
	}
}

std::string LLGridManager::getGridByProbing( const std::string &probe_for, bool case_sensitive)
{
	std::string ret;
	ret = getGridByHostName(probe_for, case_sensitive);
	if (ret.empty())
	{
		getGridByGridNick(probe_for, case_sensitive);
	}
	if (ret.empty())
	{
		getGridByLabel(probe_for, case_sensitive);
	}

	return ret;
}

std::string LLGridManager::getGridByLabel( const std::string &grid_label, bool case_sensitive)
{
	return grid_label.empty() ? std::string() : getGridByAttribute(GRID_LABEL_VALUE, grid_label, case_sensitive);
}

std::string LLGridManager::getGridByGridNick( const std::string &grid_nick, bool case_sensitive)
{
	return grid_nick.empty() ? std::string() : getGridByAttribute(GRID_NICK_VALUE, grid_nick, case_sensitive);
}

std::string LLGridManager::getGridByHostName( const std::string &host_name, bool case_sensitive)
{
	return host_name.empty() ? std::string() : getGridByAttribute(GRID_VALUE, host_name, case_sensitive);
}

std::string LLGridManager::getGridByAttribute( const std::string &attribute, const std::string &attribute_value, bool case_sensitive)
{
	if (attribute.empty() || attribute_value.empty())
	{
		return std::string();
	}

	for (LLSD::map_iterator grid_iter = mGridList.beginMap();
		grid_iter != mGridList.endMap();
		grid_iter++) 
	{
		if (grid_iter->second.has(attribute))
		{
			if (0 == (case_sensitive ? LLStringUtil::compareStrings(attribute_value, grid_iter->second[attribute].asString()) :
				LLStringUtil::compareInsensitive(attribute_value, grid_iter->second[attribute].asString())))
			{
				return grid_iter->first;
			}
		}
	}
	return std::string();
}

std::string LLGridManager::getGridId(const std::string& grid)
{
	std::string grid_id;
	std::string grid_name = getGrid(grid);
	if (!grid.empty())
	{
		// FS:AW FIRE-7468 
		// NOTE: GRID_ID_VALUE  has a different meaning in llviewernetwork 
		// than here, which was inherited from the pathfinding changes.
		// GRID_NICK_VALUE is the one we want *here*.
		grid_id = mGridList[grid_name][GRID_NICK_VALUE].asString();
	}
	else
	{
		if (LLStartUp::getStartupState() >= STATE_LOGIN_SHOW)
		{
			LL_WARNS("GridManager") << "invalid grid '" << grid << "'" << LL_ENDL;
		}
	}
	LL_DEBUGS("GridManager") << "returning " << grid_id << LL_ENDL;
	return grid_id;
}

// this assumes that there is anyway only one uri saved
std::string LLGridManager::getLoginURI(const std::string& grid)
{
	return mGridList[grid][GRID_LOGIN_URI_VALUE].beginArray()->asString();
}

void LLGridManager::getLoginURIs(std::vector<std::string>& uris)
{
	uris.clear();

	for (LLSD::array_iterator llsd_uri = mGridList[mGrid][GRID_LOGIN_URI_VALUE].beginArray();
		 llsd_uri != mGridList[mGrid][GRID_LOGIN_URI_VALUE].endArray();
		 llsd_uri++)
	{
		std::string current = llsd_uri->asString();
		if (!current.empty())
		{
			uris.push_back(current);
		}
	}
}

std::string LLGridManager::getHelperURI() 
{
	std::string cmd_line_helper_uri = gSavedSettings.getString("CmdLineHelperURI");
	if (!cmd_line_helper_uri.empty())
	{
		return cmd_line_helper_uri;
	}
	// <COLOSI opensim multi-currency support>
	std::string helperUriOverride = LFSimFeatureHandler::getInstance()->helperUriOverride();
	if (!helperUriOverride.empty())
	{
		return helperUriOverride;
	}
	// </COLOSI opensim multi-currency support>>
	return mGridList[mGrid][GRID_HELPER_URI_VALUE];
}

std::string LLGridManager::getLoginPage() 
{
	// override the login page if it was passed in
	std::string cmd_line_login_page = gSavedSettings.getString("LoginPage");
	if (!cmd_line_login_page.empty())
	{
		return cmd_line_login_page;
	}
	
	return mGridList[mGrid][GRID_LOGIN_PAGE_VALUE];
}

std::string LLGridManager::getWebProfileURL(const std::string& grid)
{
	std::string web_profile_url;
	std::string grid_name = getGrid(grid);
	if (!grid_name.empty())
	{
		web_profile_url = mGridList[grid_name][GRID_WEB_PROFILE_VALUE].asString();
	}
	else
	{
		LL_WARNS("GridManager") << "invalid grid '" << grid << "'" << LL_ENDL;
	}
	return web_profile_url;
}

void LLGridManager::updateIsInProductionGrid()
{
	EGridPlatform = GP_NOTSET;

	// *NOTE:Mani This used to compare GRID_INFO_AGNI to gGridChoice,
	// but it seems that loginURI trumps that.
	std::vector<std::string> uris;
	getLoginURIs(uris);
	if (uris.empty())
	{
		LL_DEBUGS("GridManager") << "uri is empty, setting grid platform to NOTHING." << LL_ENDL;
		return;
	}

	LLStringUtil::toLower(uris[0]);
	LLURI login_uri = LLURI(uris[0]);
	// LL looks if "agni" is contained in the string for SL main grid detection.
	// cool for http://agni.nastyfraud.com/steal.php#allyourmoney
	if (login_uri.authority().find("login.agni.lindenlab.com") ==  0)
	{
		LL_DEBUGS("GridManager")<< "uri: "<<  login_uri.authority() << " setting grid platform to SL MAIN" << LL_ENDL;
		EGridPlatform = GP_SLMAIN;
		return;
	}
	else if (login_uri.authority().find("lindenlab.com") !=  std::string::npos ) //here is no real money
	{
		LL_DEBUGS("GridManager")<< "uri: "<< login_uri.authority() << " setting grid platform to SL BETA" << LL_ENDL;
		EGridPlatform = GP_SLBETA;
		return;
	}

	if (mGridList[mGrid][GRID_PLATFORM].asString() == "Aurora")
	{
		LL_DEBUGS("GridManager")<< "uri: "<< uris[0] << "setting grid platform to AURORA" << LL_ENDL;
		EGridPlatform = GP_AURORA;
		return;
	}

	// TPVP compliance: a SL login screen must connect to SL.
	// NOTE: This is more TPVP compliant than LLs own viewer, where
	// setting the command line login page can be used for spoofing.
	LLURI login_page = LLURI(getLoginPage());
	if (login_page.authority().find("lindenlab.com") !=  std::string::npos)
	{
		setGridChoice(MAINGRID);
		return;
	}

	LL_DEBUGS("GridManager")<< "uri: "<< login_uri.authority() << " setting grid platform to OPENSIM" << LL_ENDL;
	EGridPlatform = GP_OPENSIM;
}

// For any Second Life grid
bool LLGridManager::isInSecondLife()
{
	return (EGridPlatform == GP_SLMAIN || EGridPlatform == GP_SLBETA);
}

// For Agni
bool LLGridManager::isInSLMain()
{
	return (EGridPlatform == GP_SLMAIN);
}

// For Aditi
bool LLGridManager::isInSLBeta()
{
	return (EGridPlatform == GP_SLBETA);
}

// For OpenSim
bool LLGridManager::isInOpenSim()
{
	return (EGridPlatform == GP_OPENSIM || EGridPlatform == GP_AURORA);
}

// For Aurora Sim
bool LLGridManager::isInAuroraSim()
{
	return (EGridPlatform == GP_AURORA);
}

void LLGridManager::saveGridList()
{
	// filter out just those which are not hardcoded anyway
	LLSD output_grid_list = LLSD::emptyMap();
	if (mGridList[mGrid].has("FLAG_TEMPORARY"))
	{
		mGridList[mGrid].erase("FLAG_TEMPORARY");
	}
	for(LLSD::map_iterator grid_iter = mGridList.beginMap();
		grid_iter != mGridList.endMap();
		grid_iter++)
	{
		if (!(grid_iter->first.empty() ||
			grid_iter->second.isUndefined() ||
			grid_iter->second.has("FLAG_TEMPORARY") ||
			grid_iter->second.has("DEPRECATED")))
		{
			output_grid_list[grid_iter->first] = grid_iter->second;
		}
	}
	llofstream llsd_xml;
	llsd_xml.open(mGridFile.c_str(), std::ios::out | std::ios::binary);
	LLSDSerialize::toPrettyXML(output_grid_list, llsd_xml);
	llsd_xml.close();
}

//<AW opensim>
std::string LLGridManager::trimHypergrid(const std::string& trim)
{
	std::size_t pos;
	std::string grid = trim;

	pos = grid.find_last_of(":");
	if (pos != std::string::npos)
	{
		std::string  part = grid.substr(pos+1, grid.length()-1);
		// in hope numbers only is a good guess for it's a port number
		if (std::string::npos != part.find_first_not_of("1234567890/"))
		{
			//and erase if not
			grid.erase(pos,grid.length()-1);
		}
	}

	return grid;
}
//</AW opensim>

// get location slurl base for the given region 
// within the selected grid (LL comment was misleading)
std::string LLGridManager::getSLURLBase(const std::string& grid)
{
	std::string grid_base;
	std::string ret;

	std::string grid_trimmed = trimHypergrid(grid);

	if(mGridList.has(grid_trimmed) && mGridList[grid_trimmed].has(GRID_SLURL_BASE))
	{
		ret = mGridList[grid_trimmed][GRID_SLURL_BASE].asString();
		LL_DEBUGS("GridManager") << "GRID_SLURL_BASE: " << ret << LL_ENDL;// <AW opensim>
	}
//<AW opensim>
	else
	{
		LL_DEBUGS("GridManager") << "Trying to fetch info for:" << grid << LL_ENDL;
		GridEntry* grid_entry = new GridEntry;
		grid_entry->set_current = false;
		grid_entry->grid = LLSD::emptyMap();	
		grid_entry->grid[GRID_VALUE] = grid;

		// add the grid with the additional values, or update the
		// existing grid if it exists with the given values
		addGrid(grid_entry, FETCHTEMP);

		// deal with hand edited entries
		std::string grid_norm = grid;

		if( grid_norm.size() )
		{
			size_t pos = grid_norm.find_last_of("/");

			if ( (grid_norm.length()-1) == pos )
				grid_norm.erase(pos);
		}

		ret = llformat(DEFAULT_HOP_BASE, grid_norm.c_str());// <AW: hop:// protocol>
		LL_DEBUGS("GridManager") << "DEFAULT_HOP_BASE: " << ret  << LL_ENDL;// <AW opensim>
// </AW opensim>
	}

	return  ret;
}

// get app slurl base for the given region 
// within the selected grid (LL comment was misleading)
std::string LLGridManager::getAppSLURLBase(const std::string& grid)
{
	std::string grid_base;
	std::string ret;

	if(mGridList.has(grid) && mGridList[grid].has(GRID_APP_SLURL_BASE))
	{
		ret = mGridList[grid][GRID_APP_SLURL_BASE].asString();
	}
	else
	{
		std::string app_base;
		if(mGridList.has(grid) && mGridList[grid].has(GRID_SLURL_BASE))
		{
			std::string grid_slurl_base = mGridList[grid][GRID_SLURL_BASE].asString();
			if( 0 == grid_slurl_base.find("hop://"))
			{
				app_base = DEFAULT_HOP_BASE;
				app_base.append("app");
			}
			else 
			{
				app_base = DEFAULT_APP_SLURL_BASE;
			}
		}
		else 
		{
			app_base = DEFAULT_APP_SLURL_BASE;
		}

		// deal with hand edited entries
		std::string grid_norm = grid;
		size_t pos = grid_norm.find_last_of("/");
		if ( (grid_norm.length()-1) == pos )
		{
			grid_norm.erase(pos);
		}
		ret =  llformat(app_base.c_str(), grid_norm.c_str());
	}

	LL_DEBUGS("GridManager") << "App slurl base: " << ret << " - grid = " << grid << LL_ENDL;
	return ret;
}

class FSGridManagerCommandHandler : public LLCommandHandler
{
public:
	FSGridManagerCommandHandler() : LLCommandHandler("gridmanager", UNTRUSTED_THROTTLE),
		mDownloadConnection()
	{ }

	~FSGridManagerCommandHandler()
	{
		if (mDownloadConnection.connected())
		{
			mDownloadConnection.disconnect();
		}
	}

	bool handle(const LLSD& params, const LLSD& query_map, LLMediaCtrl* web)
	{
		if (params.size() < 2)
		{
			return false;
		}

		// Automatically add and select grid via secondlife:///app/gridmanager/addgrid/<URL-encoded login URI>
		// Example: secondlife:///app/gridmanager/addgrid/http%3A%2F%2Fgrid.avatarlife.com%3A8002
		if (params[0].asString() == "addgrid")
		{
			std::string login_uri = LLURI::unescape(params[1].asString());
			mDownloadConnection = LLGridManager::instance().addGridListChangedCallback(boost::bind(&FSGridManagerCommandHandler::handleGridDownloadComplete, this, _1));
			LLGridManager::instance().addGrid(login_uri);
			return true;
		}

		return false;
	}

private:
	boost::signals2::connection mDownloadConnection;

	void handleGridDownloadComplete(bool success)
	{
		if (mDownloadConnection.connected())
		{
			mDownloadConnection.disconnect();
		}

		if (success)
		{
			LLGridManager::getInstance()->saveGridList();
			if (LLStartUp::getStartupState() <= STATE_LOGIN_WAIT)
			{
				FSPanelLogin::updateServer();
			}
		}
	}
};

// Creating the object registers with the dispatcher.
FSGridManagerCommandHandler gFSGridManagerCommandHandler;
