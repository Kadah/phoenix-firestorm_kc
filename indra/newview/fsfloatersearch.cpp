/**
 * @file fsfloatersearch.cpp
 * @brief Firestorm Search Floater
 *
 * $LicenseInfo:firstyear=2012&license=fsviewerlgpl$
 * Phoenix Firestorm Viewer Source Code
 * Copyright (C) 2012, Cinder Roxley <cinder.roxley@phoenixviewer.com>
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
 * The Phoenix Firestorm Project, Inc., 1831 Oakwood Drive, Fairmont, Minnesota 56031-3225 USA
 * http://www.firestormviewer.org
 * $/LicenseInfo$
 */

#include "llviewerprecompiledheaders.h"

#include "fsfloatersearch.h"

#include "fsavatarsearchmenu.h"
#include "fsdispatchclassifiedclickthrough.h"
#include "fsscrolllistctrl.h"
#include "lfsimfeaturehandler.h"
#include "llagent.h"
#include "llavataractions.h"
#include "llavatarname.h"
#include "llavatarnamecache.h"
#include "llavatarpropertiesprocessor.h"
#include "llclassifiedflags.h"
#include "llclassifiedinfo.h"
#include "llcombobox.h"
#include "lldateutil.h"
#include "lleventflags.h"
#include "lleventnotifier.h"
#include "llfloaterreg.h"
#include "llfloaterworldmap.h"
#include "llgroupactions.h"
#include "llgroupmgr.h"
#include "llloadingindicator.h"
#include "lllogininstance.h"
#include "llnotificationsutil.h"
#include "llpanelprofile.h"
#include "llpanelprofileclassifieds.h"
#include "llparcel.h"
#include "llproductinforequest.h"
#include "llqueryflags.h"
#include "llregionhandle.h"
#include "llremoteparcelrequest.h"
#include "lltimer.h"
#include "lltrans.h"
#include "llviewercontrol.h"
#include "llviewergenericmessage.h"
#include "llviewernetwork.h"
#include "llviewerregion.h"
#include "llworldmapmessage.h"
#include "message.h"
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <string>

static const S32 MIN_SEARCH_STRING_SIZE = 2;
static const S32 RESULT_PAGE_SIZE = 100;

std::string filterShortWords(std::string query_string);
void fillSearchComboBox(LLSearchComboBox* search_combo);

////////////////////////////////////////
//          Observer Classes          //
////////////////////////////////////////

class FSSearchRemoteParcelInfoObserver : public LLRemoteParcelInfoObserver
{
public:
	FSSearchRemoteParcelInfoObserver(FSFloaterSearch* floater, bool for_events) : LLRemoteParcelInfoObserver(),
		mParent(floater),
		mForEvents(for_events)
	{}

	~FSSearchRemoteParcelInfoObserver()
	{
		// remove any in-flight observers
		std::set<LLUUID>::iterator it;
		for (it = mParcelIDs.begin(); it != mParcelIDs.end(); ++it)
		{
			const LLUUID &id = *it;
			LLRemoteParcelInfoProcessor::getInstance()->removeObserver(id, this);
		}
		mParcelIDs.clear();
	}

	/*virtual*/ void processParcelInfo(const LLParcelData& parcel_data)
	{
		if (mParent)
		{
			if (mForEvents)
			{
				mParent->displayEventParcelImage(parcel_data);
			}
			else
			{
				mParent->displayParcelDetails(parcel_data);
			}
		}
		mParcelIDs.erase(parcel_data.parcel_id);
		LLRemoteParcelInfoProcessor::getInstance()->removeObserver(parcel_data.parcel_id, this);
	}

	/*virtual*/ void setParcelID(const LLUUID& parcel_id)
	{
		if (!parcel_id.isNull())
		{
			mParcelIDs.insert(parcel_id);
			LLRemoteParcelInfoProcessor::getInstance()->addObserver(parcel_id, this);
			LLRemoteParcelInfoProcessor::getInstance()->sendParcelInfoRequest(parcel_id);
		}
	}

	/*virtual*/ void setErrorStatus(S32 status, const std::string& reason)
	{
		LL_WARNS("Search") << "Can't complete remote parcel request. Http Status: " << status << ". Reason : " << reason << LL_ENDL;
	}
private:
	std::set<LLUUID>	mParcelIDs;
	FSFloaterSearch*	mParent;
	bool				mForEvents;
};

///// Avatar Properties Observer /////

class FSSearchAvatarPropertiesObserver : public LLAvatarPropertiesObserver
{
public:
	FSSearchAvatarPropertiesObserver(FSFloaterSearch* floater) : LLAvatarPropertiesObserver(),
	mParent(floater)
	{}

	~FSSearchAvatarPropertiesObserver()
	{
		// remove any in-flight observers
		std::set<LLUUID>::iterator it;
		for (it = mAvatarIDs.begin(); it != mAvatarIDs.end(); ++it)
		{
			const LLUUID &id = *it;
			LLAvatarPropertiesProcessor::getInstance()->removeObserver(id, this);
		}
		mAvatarIDs.clear();
	}

	void processProperties(void* data, EAvatarProcessorType type)
	{
		if (APT_PROPERTIES == type)
		{
			LLAvatarData* avatar_data = static_cast<LLAvatarData*>(data);
			if (avatar_data)
			{
				mParent->displayAvatarDetails(avatar_data);
				LLAvatarPropertiesProcessor::getInstance()->removeObserver(avatar_data->avatar_id, this);
			}
		}
		if (APT_CLASSIFIED_INFO == type)
		{
			LLAvatarClassifiedInfo* c_info = static_cast<LLAvatarClassifiedInfo*>(data);
			if (c_info)
			{
				mParent->displayClassifiedDetails(c_info);
				LLAvatarPropertiesProcessor::getInstance()->removeObserver(c_info->classified_id, this);
				std::string url = gAgent.getRegionCapability("SearchStatRequest");
				if (!url.empty())
				{
					LL_INFOS("Search") << "Classified stat request via capability" << LL_ENDL;
					LLSD body;
					body["classified_id"] = c_info->classified_id;
					LLCoreHttpUtil::HttpCoroutineAdapter::callbackHttpPost(url, body, boost::bind(&LLPanelProfileClassified::handleSearchStatResponse, c_info->classified_id, _1));
				}
			}
		}
	}
private:
	std::set<LLUUID>	mAvatarIDs;
	FSFloaterSearch*	mParent;
};

///// Group Info Observer /////

class FSSearchGroupInfoObserver : public LLGroupMgrObserver
{
public:
	FSSearchGroupInfoObserver(const LLUUID& group_id, FSFloaterSearch* parent) :
	LLGroupMgrObserver(group_id),
	mParent(parent)
	{
		LLGroupMgr* groupmgr = LLGroupMgr::getInstance();
		if (!group_id.isNull() && groupmgr)
		{
			groupmgr->addObserver(this);
			mID = group_id;
			groupmgr->sendGroupPropertiesRequest(group_id);
		}
	}

	~FSSearchGroupInfoObserver()
	{
		LLGroupMgr::getInstance()->removeObserver(this);
	}

	void changed(LLGroupChange gc)
	{
		if (gc == GC_PROPERTIES)
		{
			LLGroupMgrGroupData* group_data = LLGroupMgr::getInstance()->getGroupData(mID);
			mParent->displayGroupDetails(group_data);
			LLGroupMgr::getInstance()->removeObserver(this);
		}
	}
private:
	FSFloaterSearch*	mParent;
	LLUUID				mID;
};

///// Silly Classified Clickthrough Class /////
static FSDispatchClassifiedClickThrough sClassifiedClickThrough;

SearchQuery::SearchQuery()
: category("category", "")
, query("query")
{}

////////////////////////////////////////
//         The floater itself         //
////////////////////////////////////////

FSFloaterSearch::FSFloaterSearch(const Params& key)
:	LLFloater(key)
{
	mRemoteParcelObserver = new FSSearchRemoteParcelInfoObserver(this, false);
	mRemoteParcelEventLocationObserver = new FSSearchRemoteParcelInfoObserver(this, true);
	mAvatarPropertiesObserver = new FSSearchAvatarPropertiesObserver(this);
}

FSFloaterSearch::~FSFloaterSearch()
{
	delete mRemoteParcelObserver;
	delete mRemoteParcelEventLocationObserver;
	delete mAvatarPropertiesObserver;
	gGenericDispatcher.addHandler("classifiedclickthrough", nullptr);
}

// virtual
void FSFloaterSearch::onOpen(const LLSD& key)
{
	Params p(key);
	mPanelWeb->loadURL(p.search);
	if (key.has("query"))
	{
		mTabContainer->selectTabPanel(mPanelWeb);
	}
	else if (key.has("tab") && key["tab"].asString() == "groups")
	{
		mTabContainer->selectTabPanel(mPanelGroups);
	}

	FSSearchPanelBase* current_panel = dynamic_cast<FSSearchPanelBase*>(mTabContainer->getCurrentPanel());
	if (current_panel)
	{
		current_panel->focusDefaultElement();
	}
}

//virtual
void FSFloaterSearch::onClose(bool app_quitting)
{
	if (mTabContainer)
	{
		gSavedSettings.setS32("FSLastSearchTab", mTabContainer->getCurrentPanelIndex());
	}
}

BOOL FSFloaterSearch::postBuild()
{
	childSetAction("people_profile_btn", boost::bind(&FSFloaterSearch::onBtnPeopleProfile, this));
	childSetAction("people_message_btn", boost::bind(&FSFloaterSearch::onBtnPeopleIM, this));
	childSetAction("people_friend_btn", boost::bind(&FSFloaterSearch::onBtnPeopleFriend, this));
	childSetAction("group_profile_btn", boost::bind(&FSFloaterSearch::onBtnGroupProfile, this));
	childSetAction("group_message_btn", boost::bind(&FSFloaterSearch::onBtnGroupChat, this));
	childSetAction("group_join_btn", boost::bind(&FSFloaterSearch::onBtnGroupJoin, this));
	childSetAction("event_reminder_btn", boost::bind(&FSFloaterSearch::onBtnEventReminder, this));
	childSetAction("teleport_btn", boost::bind(&FSFloaterSearch::onBtnTeleport, this));
	childSetAction("map_btn", boost::bind(&FSFloaterSearch::onBtnMap, this));
	resetVerbs();

	mPanelPeople		= findChild<FSPanelSearchPeople>("panel_ls_people");
	mPanelGroups		= findChild<FSPanelSearchGroups>("panel_ls_groups");
	mPanelPlaces		= findChild<FSPanelSearchPlaces>("panel_ls_places");
	mPanelEvents		= findChild<FSPanelSearchEvents>("panel_ls_events");
	mPanelLand			= findChild<FSPanelSearchLand>("panel_ls_land");
	mPanelClassifieds	= findChild<FSPanelSearchClassifieds>("panel_ls_classifieds");
	mPanelWeb			= findChild<FSPanelSearchWeb>("panel_ls_web");

	mDetailsPanel =		getChild<LLPanel>("panel_ls_details");
	mDetailTitle =		getChild<LLTextEditor>("title");
	mDetailDesc =		getChild<LLTextEditor>("desc");
	mDetailAux1 =		getChild<LLTextEditor>("aux1");
	mDetailAux2 =		getChild<LLTextEditor>("aux2");
	mDetailLocation =	getChild<LLTextEditor>("location");
	mDetailSnapshot =	getChild<LLTextureCtrl>("snapshot");
	mDetailSnapshotParcel = getChild<LLTextureCtrl>("snapshot_parcel");
	mDetailMaturity =	getChild<LLIconCtrl>("maturity_icon");
	mTabContainer =		getChild<LLTabContainer>("ls_tabs");

	mTabContainer->setCommitCallback(boost::bind(&FSFloaterSearch::onTabChange, this));
	
	flushDetails();
	
	mDetailsPanel->setVisible(false);

	mHasSelection = false;

	if (!mTabContainer->selectTab(gSavedSettings.getS32("FSLastSearchTab")))
	{
		mTabContainer->selectFirstTab();
	}

	return TRUE;
}

void FSFloaterSearch::onTabChange()
{
	LLPanel* active_panel = mTabContainer->getCurrentPanel();

	if (active_panel == mPanelWeb)
	{
		mDetailsPanel->setVisible(false);
		mPanelWeb->resetFocusOnLoad();
	}
	else if (active_panel == mPanelPeople)
	{
		mDetailsPanel->setVisible(mHasSelection);
	}

	if (active_panel == mPanelPeople || active_panel == mPanelGroups)
	{
		mDetailSnapshotParcel->setVisible(FALSE);
		mDetailSnapshot->setVisible(TRUE);
	}
	else if (active_panel == mPanelPlaces || active_panel == mPanelLand ||
		active_panel == mPanelEvents || active_panel == mPanelClassifieds)
	{
		mDetailSnapshot->setVisible(FALSE);
		mDetailSnapshotParcel->setVisible(TRUE);
	}
}

//static
template <class T>
T* FSFloaterSearch::getSearchPanel(const std::string& panel_name)
{
	FSFloaterSearch* search_instance = LLFloaterReg::findTypedInstance<FSFloaterSearch>("search");
	if (search_instance && search_instance->mTabContainer)
	{
		return dynamic_cast<T*>(search_instance->mTabContainer->getPanelByName(panel_name));
	}
	else
	{
		return nullptr;
	}
}

void FSFloaterSearch::onSelectedItem(const LLUUID& selected_item, ESearchCategory type)
{
	if (!selected_item.isNull())
	{
		mSelectedID = selected_item;
		resetVerbs();
		flushDetails();
		switch (type)
		{
			case SC_AVATAR:
				LLAvatarPropertiesProcessor::getInstance()->addObserver(selected_item, mAvatarPropertiesObserver);
				LLAvatarPropertiesProcessor::getInstance()->sendAvatarPropertiesRequest(selected_item);
				break;
			case SC_GROUP:
				mGroupPropertiesRequest = new FSSearchGroupInfoObserver(selected_item, this);
				break;
			case SC_PLACE:
				mRemoteParcelObserver->setParcelID(selected_item);
				break;
			case SC_CLASSIFIED:
				LLAvatarPropertiesProcessor::getInstance()->addObserver(selected_item, mAvatarPropertiesObserver);
				LLAvatarPropertiesProcessor::getInstance()->sendClassifiedInfoRequest(selected_item);
				gGenericDispatcher.addHandler("classifiedclickthrough", &sClassifiedClickThrough);
				break;
		}
		setLoadingProgress(true);
	}
}

void FSFloaterSearch::onSelectedEvent(const S32 selected_event)
{
	resetVerbs();
	flushDetails();

	gMessageSystem->newMessageFast(_PREHASH_EventInfoRequest);
	gMessageSystem->nextBlockFast(_PREHASH_AgentData);
	gMessageSystem->addUUIDFast(_PREHASH_AgentID, gAgentID);
	gMessageSystem->addUUIDFast(_PREHASH_SessionID, gAgentSessionID);
	gMessageSystem->nextBlockFast(_PREHASH_EventData);
	gMessageSystem->addU32Fast(_PREHASH_EventID, selected_event);
	gAgent.sendReliableMessage();
}

void FSFloaterSearch::displayParcelDetails(const LLParcelData& parcel_data)
{
	S32 region_x;
	S32 region_y;
	S32 region_z;
	region_x = ll_round(parcel_data.global_x) % REGION_WIDTH_UNITS;
	region_y = ll_round(parcel_data.global_y) % REGION_WIDTH_UNITS;
	region_z = ll_round(parcel_data.global_z);
	// HACK: Flag 0x2 == adult region,
	// Flag 0x1 == mature region, otherwise assume PG
	if (parcel_data.flags & 0x2)
	{
		mDetailMaturity->setValue("Parcel_R_Dark");
	}
	else if (parcel_data.flags & 0x1)
	{
		mDetailMaturity->setValue("Parcel_M_Dark");
	}
	else
	{
		mDetailMaturity->setValue("Parcel_PG_Dark");
	}

	LLStringUtil::format_map_t map;
	map["DWELL"] = llformat("%.0f", (F64)parcel_data.dwell);
	map["AREA"] = llformat("%d m²", parcel_data.actual_area);
	map["LOCATION"] = llformat("%s (%d, %d, %d)", parcel_data.sim_name.c_str(), region_x, region_y, region_z);

	mParcelGlobal = LLVector3d(parcel_data.global_x, parcel_data.global_y, parcel_data.global_z);
	mDetailsPanel->setVisible(mTabContainer->getCurrentPanel()->getName() == "panel_ls_places" || mTabContainer->getCurrentPanel()->getName() == "panel_ls_land");
	mHasSelection = true;
	mDetailMaturity->setVisible(true);
	mDetailTitle->setValue(parcel_data.name);
	mDetailDesc->setValue(parcel_data.desc);
	mDetailAux1->setValue(getString("string.traffic", map));
	mDetailAux2->setValue(getString("string.area", map));
	mDetailLocation->setValue(getString("string.location", map));
	mDetailSnapshotParcel->setValue(parcel_data.snapshot_id);
	childSetVisible("teleport_btn", true);
	childSetVisible("map_btn", true);
	setLoadingProgress(false);
}

void FSFloaterSearch::displayAvatarDetails(LLAvatarData*& avatar_data)
{
	if (avatar_data)
	{
		LLStringUtil::format_map_t map;
		map["AGE"] = LLDateUtil::ageFromDate(avatar_data->born_on, LLDate::now());
		if (avatar_data->partner_id.notNull())
		{
			map["PARTNER"] = LLSLURL("agent", avatar_data->partner_id, "inspect").getSLURLString();
			mDetailAux2->setValue(getString("string.partner", map));
		}

		mDetailsPanel->setVisible(mTabContainer->getCurrentPanel()->getName() == "panel_ls_people");
		mHasSelection = true;
		mDetailTitle->setValue(LLTrans::getString("LoadingData"));
		mDetailDesc->setValue(avatar_data->about_text);
		mDetailSnapshot->setValue(avatar_data->image_id);
		mDetailAux1->setValue(getString("string.age", map));
		LLAvatarNameCache::get(avatar_data->avatar_id, boost::bind(&FSFloaterSearch::avatarNameUpdatedCallback,this, _1, _2));
		childSetVisible("people_profile_btn", true);
		childSetVisible("people_message_btn", true);
		childSetVisible("people_friend_btn", true);
		getChildView("people_friend_btn")->setEnabled(!LLAvatarActions::isFriend(avatar_data->avatar_id));
	}
}

void FSFloaterSearch::displayGroupDetails(LLGroupMgrGroupData*& group_data)
{
	if (group_data)
	{
		LLStringUtil::format_map_t map;
		map["MEMBER_COUNT"] = llformat("%d",group_data->mMemberCount);
		map["FOUNDER"] = LLSLURL("agent", group_data->mFounderID, "inspect").getSLURLString();

		mDetailsPanel->setVisible(mTabContainer->getCurrentPanel()->getName() == "panel_ls_groups");
		mHasSelection = true;
		mDetailTitle->setValue(LLTrans::getString("LoadingData"));
		mDetailDesc->setValue(group_data->mCharter);
		mDetailSnapshot->setValue(group_data->mInsigniaID);
		mDetailAux1->setValue(getString("string.members", map));
		mDetailAux2->setValue(getString("string.founder", map));
		LLGroupData agent_gdatap;
		bool is_member = gAgent.getGroupData(getSelectedID(),agent_gdatap) || gAgent.isGodlike();
		bool join_btn_enabled = !is_member && group_data->mOpenEnrollment;
		childSetVisible("group_profile_btn", true);
		childSetVisible("group_message_btn", true);
		childSetVisible("group_join_btn", true);
		getChildView("group_join_btn")->setEnabled(join_btn_enabled);
		getChildView("group_message_btn")->setEnabled(is_member);
		gCacheName->getGroup(getSelectedID(), boost::bind(&FSFloaterSearch::groupNameUpdatedCallback, this, _1, _2, _3));
	}
}

void FSFloaterSearch::displayClassifiedDetails(LLAvatarClassifiedInfo*& c_info)
{
	if (c_info)
	{
		if (c_info->flags & CLASSIFIED_FLAG_MATURE)
		{
			mDetailMaturity->setValue("Parcel_M_Dark");
		}
		else
		{
			mDetailMaturity->setValue("Parcel_PG_Dark");
		}

		LLStringUtil::format_map_t map;
		map["LISTING_PRICE"] = llformat("L$%d", c_info->price_for_listing);
		map["SLURL"] = LLSLURL("parcel", c_info->parcel_id, "about").getSLURLString();

		mDetailsPanel->setVisible(mTabContainer->getCurrentPanel()->getName() == "panel_ls_classifieds");
		mHasSelection = true;
		mDetailMaturity->setVisible(true);
		mParcelGlobal = c_info->pos_global;
		mDetailTitle->setValue(c_info->name);
		mDetailDesc->setValue(c_info->description);
		mDetailSnapshotParcel->setValue(c_info->snapshot_id);
		mDetailAux1->setValue(getString("string.listing_price", map));
		mDetailLocation->setValue(getString("string.slurl", map));
		childSetVisible("teleport_btn", true);
		childSetVisible("map_btn", true);
		setLoadingProgress(false);
	}
}

void FSFloaterSearch::displayEventDetails(U32 eventId, F64 eventEpoch, const std::string& eventDateStr, const std::string &eventName, const std::string &eventDesc, const std::string &simName, U32 eventDuration, U32 eventFlags, U32 eventCover, LLVector3d eventGlobalPos)
{
	if (eventFlags == EVENT_FLAG_ADULT)
	{
		mDetailMaturity->setValue("Parcel_R_Dark");
	}
	else if (eventFlags == EVENT_FLAG_MATURE)
	{
		mDetailMaturity->setValue("Parcel_M_Dark");
	}
	else
	{
		mDetailMaturity->setValue("Parcel_PG_Dark");
	}

	S32 region_x;
	S32 region_y;
	S32 region_z;
	region_x = (S64)ll_round(eventGlobalPos.mdV[VX]) % REGION_WIDTH_UNITS;
	region_y = (S64)ll_round(eventGlobalPos.mdV[VY]) % REGION_WIDTH_UNITS;
	region_z = (S32)ll_round(eventGlobalPos.mdV[VZ]);
	LLStringUtil::format_map_t map;
	map["DURATION"] = llformat("%d:%.2d", eventDuration / 60, eventDuration % 60);
	map["LOCATION"] = llformat("%s (%d, %d, %d)", simName.c_str(), region_x, region_y, region_z);
	if (eventCover > 0)
	{
		map["COVERCHARGE"] = llformat("L$%d", eventCover);
		mDetailAux2->setValue(getString("string.covercharge", map));
	}

	mParcelGlobal = eventGlobalPos;
	mEventID = eventId;
	mDetailsPanel->setVisible(mTabContainer->getCurrentPanel()->getName() == "panel_ls_events");
	mHasSelection = true;
	mDetailMaturity->setVisible(true);
	mDetailTitle->setValue(eventName);
	mDetailDesc->setValue(eventDesc);
	mDetailAux1->setValue(getString("string.duration", map));
	mDetailLocation->setValue(getString("string.location", map));
	mDetailSnapshotParcel->setValue(LLUUID::null);
	childSetVisible("teleport_btn", true);
	childSetVisible("map_btn", true);
	childSetVisible("event_reminder_btn", true);

	LLWorldMapMessage::getInstance()->sendNamedRegionRequest(simName, boost::bind(&FSFloaterSearch::regionHandleCallback, this, _1, eventGlobalPos), "", false);
}

void FSFloaterSearch::regionHandleCallback(U64 region_handle, LLVector3d pos_global)
{
	std::string url = gAgent.getRegionCapability("RemoteParcelRequest");
	if (!url.empty())
	{
		auto region_origin = from_region_handle(region_handle);
		LLVector3 pos_region(LLVector3(pos_global - region_origin));

		LLRemoteParcelInfoProcessor::getInstance()->requestRegionParcelInfo(url,
			LLUUID::null, pos_region, pos_global, mRemoteParcelEventLocationObserver->getObserverHandle());
	}
	else
	{
		setLoadingProgress(false);
	}
}

void FSFloaterSearch::displayEventParcelImage(const LLParcelData& parcel_data)
{
	mDetailSnapshotParcel->setValue(parcel_data.snapshot_id);
	setLoadingProgress(false);
}

void FSFloaterSearch::avatarNameUpdatedCallback(const LLUUID& id, const LLAvatarName& av_name)
{
	if (id == getSelectedID())
	{
		mDetailTitle->setValue(av_name.getCompleteName());
		setLoadingProgress(false);
	}
	// Otherwise possibly a request for an older selection, ignore it.
}

void FSFloaterSearch::groupNameUpdatedCallback(const LLUUID& id, const std::string& name, bool is_group)
{
	if (id == getSelectedID())
	{
		mDetailTitle->setValue( LLSD(name) );
		setLoadingProgress(false);
	}
	// Otherwise possibly a request for an older selection, ignore it.
}

void FSFloaterSearch::setLoadingProgress(bool started)
{
	LLLoadingIndicator* indicator = getChild<LLLoadingIndicator>("loading");

	indicator->setVisible(started);

	if (started)
	{
		indicator->start();
	}
	else
	{
		indicator->stop();
	}
}

void FSFloaterSearch::resetVerbs()
{
	childSetVisible("people_profile_btn", false);
	childSetVisible("people_message_btn", false);
	childSetVisible("people_friend_btn", false);
	childSetVisible("group_profile_btn", false);
	childSetVisible("group_message_btn", false);
	childSetVisible("group_join_btn", false);
	childSetVisible("event_reminder_btn", false);
	childSetVisible("teleport_btn", false);
	childSetVisible("map_btn", false);
}

void FSFloaterSearch::flushDetails()
{
	mDetailTitle->setValue("");
	mDetailDesc->setValue("");
	mDetailAux1->setValue("");
	mDetailAux2->setValue("");
	mDetailLocation->setValue("");
	mDetailSnapshot->setValue(LLSD());
	mDetailMaturity->setVisible(false);
	mParcelGlobal.setZero();
}

void FSFloaterSearch::onBtnPeopleProfile()
{
	LLAvatarActions::showProfile(getSelectedID());
}

void FSFloaterSearch::onBtnPeopleIM()
{
	LLAvatarActions::startIM(getSelectedID());
}

void FSFloaterSearch::onBtnPeopleFriend()
{
	LLAvatarActions::requestFriendshipDialog(getSelectedID());
}

void FSFloaterSearch::onBtnGroupProfile()
{
	LLGroupActions::show(getSelectedID());
}

void FSFloaterSearch::onBtnGroupChat()
{
	LLGroupActions::startIM(getSelectedID());
}

void FSFloaterSearch::onBtnGroupJoin()
{
	LLGroupActions::join(getSelectedID());
}

void FSFloaterSearch::onBtnTeleport()
{
	if (!mParcelGlobal.isExactlyZero())
	{
		gAgent.teleportViaLocation(mParcelGlobal);
		LLFloaterWorldMap::getInstance()->trackLocation(mParcelGlobal);
		/// <FS:CR> What should we do when when we teleport? The default (1) is to close the floater,
		/// the user may elect to minimize the floater (2), or to do nothing (any other setting)
		static LLCachedControl<U32> teleport_action(gSavedSettings, "FSLegacySearchActionOnTeleport");
		if (teleport_action == 1)
		{
			closeFloater();
		}
		else if (teleport_action == 2)
		{
			setMinimized(TRUE);
		}
	}
}

void FSFloaterSearch::onBtnMap()
{
	if (!mParcelGlobal.isExactlyZero())
	{
		LLFloaterWorldMap::getInstance()->trackLocation(mParcelGlobal);
		LLFloaterReg::showInstance("world_map", "center");
	}
}

void FSFloaterSearch::onBtnEventReminder()
{
	gEventNotifier.add(mEventID);
}

////////////////////////////////////////
//         People Search Panel        //
////////////////////////////////////////

static LLPanelInjector<FSPanelSearchPeople> t_panel_fs_search_people("panel_ls_people");

FSPanelSearchPeople::FSPanelSearchPeople() : FSSearchPanelBase()
, mQueryID(nullptr)
, mStartSearch(0)
, mResultsReceived(0)
, mResultsContent()
, mAvatarNameCallbackConnection()
{
}

FSPanelSearchPeople::~FSPanelSearchPeople()
{
	if (mAvatarNameCallbackConnection.connected())
	{
		mAvatarNameCallbackConnection.disconnect();
	}
}

BOOL FSPanelSearchPeople::postBuild()
{
	mSearchComboBox =	findChild<LLSearchComboBox>("people_edit");
	mSearchResults =	findChild<FSScrollListCtrl>("search_results_people");
	if (mSearchComboBox)
	{
		mSearchComboBox->setCommitCallback(boost::bind(&FSPanelSearchPeople::onBtnFind, this));
		fillSearchComboBox(mSearchComboBox);
	}
	if (mSearchResults)
	{
		mSearchResults->setCommitCallback(boost::bind(&FSPanelSearchPeople::onSelectItem, this));
		mSearchResults->setEnabled(FALSE);
		mSearchResults->setCommentText(LLTrans::getString("no_results"));
		mSearchResults->setContextMenu(&gFSAvatarSearchMenu);
	}

	childSetAction("people_next", boost::bind(&FSPanelSearchPeople::onBtnNext, this));
	childSetAction("people_back", boost::bind(&FSPanelSearchPeople::onBtnBack, this));
	getChildView("people_next")->setEnabled(FALSE);
	getChildView("people_back")->setEnabled(FALSE);

	return TRUE;
}

void FSPanelSearchPeople::focusDefaultElement()
{
	mSearchComboBox->focusTextEntry();
}

void FSPanelSearchPeople::find()
{
	std::string text = mSearchComboBox->getSimple();
	boost::trim(text);
	if (text.size() <= MIN_SEARCH_STRING_SIZE)
	{
		mSearchResults->setCommentText(LLTrans::getString("search_short"));
		return;
	}

	if (LLUUID::validate(text))
	{
		LLUUID id(text);

		mSearchResults->deleteAllItems();
		mSearchResults->setCommentText(LLTrans::getString("searching"));
		mResultsReceived = 0;
		mNumResultsReturned = 0;

		if (mAvatarNameCallbackConnection.connected())
		{
			mAvatarNameCallbackConnection.disconnect();
		}
		mAvatarNameCallbackConnection = LLAvatarNameCache::get(id, boost::bind(&FSPanelSearchPeople::onAvatarNameCallback, this, _1, _2));

		return;
	}

	LLStringUtil::replaceChar(text, '.', ' ');

	mResultsReceived = 0;
	if (mQueryID.notNull())
	{
		mQueryID.setNull();
	}
	mQueryID.generate();

	if (mStartSearch < 0)
	{
		mStartSearch = 0;
	}

	gMessageSystem->newMessage("DirFindQuery");
	gMessageSystem->nextBlock("AgentData");
	gMessageSystem->addUUID("AgentID", gAgentID);
	gMessageSystem->addUUID("SessionID", gAgentSessionID);
	gMessageSystem->nextBlock("QueryData");
	gMessageSystem->addUUID("QueryID", getQueryID());
	gMessageSystem->addString("QueryText", text);
	gMessageSystem->addU32("QueryFlags", DFQ_PEOPLE);
	gMessageSystem->addS32("QueryStart", mStartSearch);
	gAgent.sendReliableMessage();
	LL_DEBUGS("Search") << "Firing off search request: " << getQueryID() << LL_ENDL;

	mSearchResults->deleteAllItems();
	mSearchResults->setCommentText(LLTrans::getString("searching"));
	mNumResultsReturned = 0;
}

void FSPanelSearchPeople::onBtnFind()
{
	std::string text = mSearchComboBox->getSimple();
	if (!text.empty())
	{
		LLSearchHistory::getInstance()->addEntry(text);
	}

	resetSearch();
	
	find();
}

void FSPanelSearchPeople::onBtnNext()
{
	mStartSearch += RESULT_PAGE_SIZE;
	getChildView("people_back")->setEnabled(TRUE);

	find();
}

void FSPanelSearchPeople::onBtnBack()
{
	mStartSearch -= RESULT_PAGE_SIZE;
	getChildView("people_back")->setEnabled(mStartSearch > 0);

	find();
}

void FSPanelSearchPeople::resetSearch()
{
	mStartSearch = 0;
	getChildView("people_back")->setEnabled(FALSE);
	getChildView("people_next")->setEnabled(FALSE);
}

S32 FSPanelSearchPeople::showNextButton(S32 rows)
{
	bool show_next_button = (mResultsReceived > RESULT_PAGE_SIZE);
	getChildView("people_next")->setEnabled(show_next_button);
	if (show_next_button)
	{
		rows -= (mResultsReceived - RESULT_PAGE_SIZE);
	}
	return rows;
}

void FSPanelSearchPeople::onSelectItem()
{
	if (!mSearchResults)
	{
		return;
	}
	FSFloaterSearch* search_instance = LLFloaterReg::findTypedInstance<FSFloaterSearch>("search");
	if (search_instance)
	{
		search_instance->FSFloaterSearch::onSelectedItem(mSearchResults->getSelectedValue(), FSFloaterSearch::SC_AVATAR);
	}
}

// static
void FSPanelSearchPeople::processSearchReply(LLMessageSystem* msg, void**)
{
	LLUUID query_id;
	std::string first_name;
	std::string last_name;
	LLUUID agent_id;

	msg->getUUIDFast(_PREHASH_QueryData,	_PREHASH_QueryID,	query_id);
	msg->getUUIDFast(_PREHASH_AgentData,	_PREHASH_AgentID,	agent_id);

	// This result is not for us.
	if (agent_id != gAgentID)
	{
		return;
	}
	LL_DEBUGS("Search") << "received search results - QueryID: " << query_id << " AgentID: " << agent_id << LL_ENDL;

	FSPanelSearchPeople* self = FSFloaterSearch::getSearchPanel<FSPanelSearchPeople>("panel_ls_people");

	// floater is closed or these are not results from our last request
	if (!self || query_id != self->getQueryID())
	{
		return;
	}

	LLScrollListCtrl* search_results = self->getChild<LLScrollListCtrl>("search_results_people");

	if (self->mNumResultsReturned++ == 0)
	{
		search_results->deleteAllItems();
	}

	// Check for status messages
	if (msg->getNumberOfBlocks("StatusData"))
	{
		U32 status;
		msg->getU32("StatusData", "Status", status);
		if (status & STATUS_SEARCH_PLACES_FOUNDNONE)
		{
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("people_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_SHORTSTRING)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_short"));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_BANNEDWORD)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_banned"));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_SEARCHDISABLED)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_disabled"));
			return;
		}
	}

	bool found_one = false;
	S32 num_new_rows = msg->getNumberOfBlocksFast(_PREHASH_QueryReplies);
	if (num_new_rows == 0 && self->mResultsReceived == 0)
	{
		LLStringUtil::format_map_t map;
		map["[TEXT]"] = self->getChild<LLUICtrl>("people_edit")->getValue().asString();
		search_results->setEnabled(FALSE);
		search_results->setCommentText(LLTrans::getString("not_found", map));
	}

	self->mResultsReceived += num_new_rows;
	num_new_rows = self->showNextButton(num_new_rows);

	for (S32 i = 0; i < num_new_rows; i++)
	{
		msg->getStringFast(	_PREHASH_QueryReplies,	_PREHASH_FirstName,	first_name, i);
		msg->getStringFast(	_PREHASH_QueryReplies,	_PREHASH_LastName,	last_name, i);
		msg->getUUIDFast(	_PREHASH_QueryReplies,	_PREHASH_AgentID,	agent_id, i);
		//msg->getU8Fast(	_PREHASH_QueryReplies,	_PREHASH_Online,	online, i);
		
		if (agent_id.isNull())
		{
			LL_INFOS("Search") << "Null result returned for QueryID: " << query_id << LL_ENDL;
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("people_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
		}
		else
		{
			LL_DEBUGS("Search") << "Got: " << first_name << " " << last_name << " AgentID: " << agent_id << LL_ENDL;
			search_results->setEnabled(TRUE);
			found_one = true;
			
			std::string avatar_name;
			avatar_name = LLCacheName::buildFullName(first_name, last_name);

			LLSD content;
			LLSD element;

			element["id"] = agent_id;

			element["columns"][0]["column"]	= "icon";
			element["columns"][0]["type"]	= "icon";
			element["columns"][0]["value"]	= "icon_avatar_offline.tga";

			element["columns"][1]["column"]	= "username";
			element["columns"][1]["value"]	= avatar_name;

			content["name"] = avatar_name;

			search_results->addElement(element, ADD_BOTTOM);
			self->mResultsContent[agent_id.asString()] = content;
		}
	}
	if (found_one)
	{
		search_results->selectFirstItem();
		search_results->setFocus(TRUE);
		self->onSelectItem();
	}
}

void FSPanelSearchPeople::onAvatarNameCallback(const LLUUID& id, const LLAvatarName& av_name)
{
	if (mAvatarNameCallbackConnection.connected())
	{
		mAvatarNameCallbackConnection.disconnect();
	}

	LLScrollListCtrl* search_results = getChild<LLScrollListCtrl>("search_results_people");

	if (av_name.getAccountName() != "(?\?\?).(?\?\?)")
	{
		LLSD content;
		LLSD data;
		data["id"] = id;

		data["columns"][0]["column"] = "icon";
		data["columns"][0]["type"] = "icon";
		data["columns"][0]["value"] = "icon_avatar_offline.tga";

		data["columns"][1]["name"] = "username";
		data["columns"][1]["value"] = av_name.getUserName();

		content["name"] = av_name.getUserName();

		search_results->addElement(data);

		mResultsContent[id.asString()] = content;
		mResultsReceived = 1;
		mNumResultsReturned = 1;

		search_results->setEnabled(TRUE);
		search_results->selectFirstItem();
		search_results->setFocus(TRUE);
		onSelectItem();
	}
	else
	{
		LLStringUtil::format_map_t map;
		map["[TEXT]"] = getChild<LLUICtrl>("people_edit")->getValue().asString();
		search_results->setEnabled(FALSE);
		search_results->setCommentText(LLTrans::getString("not_found", map));
	}
}

////////////////////////////////////////
//         Groups Search Panel        //
////////////////////////////////////////

static LLPanelInjector<FSPanelSearchGroups> t_panel_fs_search_groups("panel_ls_groups");

FSPanelSearchGroups::FSPanelSearchGroups() : FSSearchPanelBase()
, mQueryID(nullptr)
, mStartSearch(0)
, mResultsReceived(0)
, mResultsContent()
{
}

FSPanelSearchGroups::~FSPanelSearchGroups()
{
}

BOOL FSPanelSearchGroups::postBuild()
{
	mSearchComboBox =	findChild<LLSearchComboBox>("groups_edit");
	mSearchResults =	findChild<LLScrollListCtrl>("search_results_groups");
	if (mSearchComboBox)
	{
		mSearchComboBox->setCommitCallback(boost::bind(&FSPanelSearchGroups::onBtnFind, this));
		fillSearchComboBox(mSearchComboBox);
	}
	if (mSearchResults)
	{
		mSearchResults->setCommitCallback(boost::bind(&FSPanelSearchGroups::onSelectItem, this));
		mSearchResults->setEnabled(FALSE);
		mSearchResults->setCommentText(LLTrans::getString("no_results"));
	}

	childSetAction("groups_next", boost::bind(&FSPanelSearchGroups::onBtnNext, this));
	childSetAction("groups_back", boost::bind(&FSPanelSearchGroups::onBtnBack, this));
	getChildView("groups_next")->setEnabled(FALSE);
	getChildView("groups_back")->setEnabled(FALSE);

	return TRUE;
}

void FSPanelSearchGroups::focusDefaultElement()
{
	mSearchComboBox->focusTextEntry();
}

void FSPanelSearchGroups::find()
{
	std::string text = filterShortWords(mSearchComboBox->getSimple());
	if (text.size() == 0)
	{
		mSearchResults->setCommentText(LLTrans::getString("search_short"));
		return;
	}

	static LLUICachedControl<bool> inc_pg("ShowPGSims", 1);
	static LLUICachedControl<bool> inc_mature("ShowMatureSims", 0);
	static LLUICachedControl<bool> inc_adult("ShowAdultSims", 0);
	if (!(inc_pg || inc_mature || inc_adult))
	{
		LLNotificationsUtil::add("NoContentToSearch");
		return;
	}
	U32 scope = 0;
	if (gAgent.wantsPGOnly())
	{
		scope |= DFQ_PG_SIMS_ONLY;
	}
	bool adult_enabled = gAgent.canAccessAdult();
	bool mature_enabled = gAgent.canAccessMature();
	if (inc_pg)
	{
		scope |= DFQ_INC_PG;
	}
	if (inc_mature && mature_enabled)
	{
		scope |= DFQ_INC_MATURE;
	}
	if (inc_adult && adult_enabled)
	{
		scope |= DFQ_INC_ADULT;
	}
	scope |= DFQ_GROUPS;

	mResultsReceived = 0;
	if (mQueryID.notNull())
	{
		mQueryID.setNull();
	}
	mQueryID.generate();

	if (mStartSearch < 0)
	{
		mStartSearch = 0;
	}

	gMessageSystem->newMessage("DirFindQuery");
	gMessageSystem->nextBlock("AgentData");
	gMessageSystem->addUUID("AgentID", gAgentID);
	gMessageSystem->addUUID("SessionID", gAgentSessionID);
	gMessageSystem->nextBlock("QueryData");
	gMessageSystem->addUUID("QueryID", getQueryID());
	gMessageSystem->addString("QueryText", text);
	gMessageSystem->addU32("QueryFlags", scope);
	gMessageSystem->addS32("QueryStart", mStartSearch);
	gAgent.sendReliableMessage();
	LL_DEBUGS("Search") << "Firing off search request: " << getQueryID() << LL_ENDL;

	mSearchResults->deleteAllItems();
	mSearchResults->setCommentText(LLTrans::getString("searching"));
	mNumResultsReturned = 0;
}

void FSPanelSearchGroups::onBtnFind()
{
	std::string text = mSearchComboBox->getSimple();
	if (!text.empty())
	{
		LLSearchHistory::getInstance()->addEntry(text);
	}

	resetSearch();
	
	find();
}

void FSPanelSearchGroups::onBtnNext()
{
	mStartSearch += RESULT_PAGE_SIZE;
	getChildView("groups_back")->setEnabled(TRUE);

	find();
}

void FSPanelSearchGroups::onBtnBack()
{
	mStartSearch -= RESULT_PAGE_SIZE;
	getChildView("groups_back")->setEnabled(mStartSearch > 0);

	find();
}

void FSPanelSearchGroups::resetSearch()
{
	mStartSearch = 0;
	getChildView("groups_back")->setEnabled(FALSE);
	getChildView("groups_next")->setEnabled(FALSE);
}

S32 FSPanelSearchGroups::showNextButton(S32 rows)
{
	bool show_next_button = (mResultsReceived > RESULT_PAGE_SIZE);
	getChildView("groups_next")->setEnabled(show_next_button);
	if (show_next_button)
	{
		rows -= (mResultsReceived - RESULT_PAGE_SIZE);
	}
	return rows;
}

void FSPanelSearchGroups::onSelectItem()
{
	if (!mSearchResults)
	{
		return;
	}
	FSFloaterSearch* search_instance = LLFloaterReg::findTypedInstance<FSFloaterSearch>("search");
	if (search_instance)
	{
		search_instance->FSFloaterSearch::onSelectedItem(mSearchResults->getSelectedValue(), FSFloaterSearch::SC_GROUP);
	}
}

// static
void FSPanelSearchGroups::processSearchReply(LLMessageSystem* msg, void**)
{
	LLUUID query_id;
	LLUUID group_id;
	LLUUID agent_id;
	std::string group_name;
	S32 members;
	F32 search_order;

	msg->getUUIDFast(	_PREHASH_QueryData,	_PREHASH_QueryID,	query_id);
	msg->getUUIDFast(	_PREHASH_AgentData,	_PREHASH_AgentID,	agent_id);

	// Not for us
	if (agent_id != gAgentID)
	{
		return;
	}
	LL_DEBUGS("Search") << "received directory request - QueryID: " << query_id << " AgentID: " << agent_id << LL_ENDL;

	FSPanelSearchGroups* self = FSFloaterSearch::getSearchPanel<FSPanelSearchGroups>("panel_ls_groups");

	// floater is closed or these are not results from our last request
	if (!self || query_id != self->mQueryID)
	{
		return;
	}

	LLScrollListCtrl* search_results = self->getChild<LLScrollListCtrl>("search_results_groups");

	// Clear "Searching" label on first results
	if (self->mNumResultsReturned++ == 0)
	{
		search_results->deleteAllItems();
	}

	// Check for status messages
	if (msg->getNumberOfBlocks("StatusData"))
	{
		U32 status;
		msg->getU32("StatusData", "Status", status);
		if (status & STATUS_SEARCH_PLACES_FOUNDNONE)
		{
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("groups_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
			return;
		}
		else if(status & STATUS_SEARCH_PLACES_SHORTSTRING)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_short"));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_BANNEDWORD)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_banned"));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_SEARCHDISABLED)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_disabled"));
			return;
		}
	}

	bool found_one = false;
	S32 num_new_rows = msg->getNumberOfBlocksFast(_PREHASH_QueryReplies);
	if (num_new_rows == 0 && self->mResultsReceived == 0)
	{
		LLStringUtil::format_map_t map;
		map["[TEXT]"] = self->getChild<LLUICtrl>("groups_edit")->getValue().asString();
		search_results->setEnabled(FALSE);
		search_results->setCommentText(LLTrans::getString("not_found", map));
	}

	self->mResultsReceived += num_new_rows;
	num_new_rows = self->showNextButton(num_new_rows);

	for (S32 i = 0; i < num_new_rows; i++)
	{
		msg->getUUIDFast(	_PREHASH_QueryReplies,	_PREHASH_GroupID,		group_id,	i);
		msg->getStringFast(	_PREHASH_QueryReplies,	_PREHASH_GroupName,		group_name,	i);
		msg->getS32Fast(	_PREHASH_QueryReplies,	_PREHASH_Members,		members,	i);
		msg->getF32Fast(	_PREHASH_QueryReplies,	_PREHASH_SearchOrder,	search_order,i);
		if (group_id.isNull())
		{
			LL_DEBUGS("Search") << "No results returned for QueryID: " << query_id << LL_ENDL;
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("groups_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
		}
		else
		{
			LL_DEBUGS("Search") << "Got: " << group_name << " GroupID: " << group_id << LL_ENDL;
			search_results->setEnabled(TRUE);
			found_one = true;

			LLSD content;
			LLSD element;

			element["id"] = group_id;

			element["columns"][0]["column"]	= "icon";
			element["columns"][0]["type"]	= "icon";
			element["columns"][0]["value"]	= "Icon_Group";

			element["columns"][1]["column"]	= "group_name";
			element["columns"][1]["value"]	= group_name;

			element["columns"][2]["column"]	= "members";
			element["columns"][2]["value"]	= members;

			element["columns"][3]["column"]	= "score";
			element["columns"][3]["value"]	= search_order;

			content["name"] = group_name;

			search_results->addElement(element, ADD_BOTTOM);
			self->mResultsContent[group_id.asString()] = content;
		}
	}
	if (found_one)
	{
		search_results->selectFirstItem();
		search_results->setFocus(TRUE);
		self->onSelectItem();
	}
}

////////////////////////////////////////
//         Places Search Panel        //
////////////////////////////////////////

static LLPanelInjector<FSPanelSearchPlaces> t_panel_fs_search_places("panel_ls_places");

FSPanelSearchPlaces::FSPanelSearchPlaces() : FSSearchPanelBase()
, mQueryID(nullptr)
, mStartSearch(0)
, mResultsReceived(0)
, mResultsContent()
{
	mCommitCallbackRegistrar.add("CommitSearch", boost::bind(&FSPanelSearchPlaces::find, this));
}

FSPanelSearchPlaces::~FSPanelSearchPlaces()
{
}

BOOL FSPanelSearchPlaces::postBuild()
{
	mSearchComboBox =	findChild<LLSearchComboBox>("places_edit");
	mSearchResults =	findChild<LLScrollListCtrl>("search_results_places");
	mPlacesCategory =	findChild<LLComboBox>("places_category");
	if (mSearchComboBox)
	{
		mSearchComboBox->setCommitCallback(boost::bind(&FSPanelSearchPlaces::onBtnFind, this));
		fillSearchComboBox(mSearchComboBox);
	}
	if (mSearchResults)
	{
		mSearchResults->setCommitCallback(boost::bind(&FSPanelSearchPlaces::onSelectItem, this));
		mSearchResults->setEnabled(FALSE);
		mSearchResults->setCommentText(LLTrans::getString("no_results"));
	}
	if (mPlacesCategory)
	{
		mPlacesCategory->add(LLTrans::getString("all_categories"), LLSD("any"));
		mPlacesCategory->addSeparator();
		for (int category = LLParcel::C_LINDEN; category < LLParcel::C_COUNT; category++)
		{
			LLParcel::ECategory eCategory = (LLParcel::ECategory)category;
			mPlacesCategory->add(LLTrans::getString(LLParcel::getCategoryUIString(eCategory)), LLParcel::getCategoryString(eCategory));
		}
	}
	childSetAction("places_next", boost::bind(&FSPanelSearchPlaces::onBtnNext, this));
	childSetAction("places_back", boost::bind(&FSPanelSearchPlaces::onBtnBack, this));
	getChildView("places_next")->setEnabled(FALSE);
	getChildView("places_back")->setEnabled(FALSE);

	return TRUE;
}

void FSPanelSearchPlaces::focusDefaultElement()
{
	mSearchComboBox->focusTextEntry();
}

void FSPanelSearchPlaces::find()
{
	std::string text = filterShortWords(mSearchComboBox->getSimple());
	if (text.size() == 0)
	{
		mSearchResults->setCommentText(LLTrans::getString("search_short"));
		return;
	}

	static LLUICachedControl<bool> inc_pg("ShowPGSims", 1);
	static LLUICachedControl<bool> inc_mature("ShowMatureSims", 0);
	static LLUICachedControl<bool> inc_adult("ShowAdultSims", 0);
	if (!(inc_pg || inc_mature || inc_adult))
	{
		LLNotificationsUtil::add("NoContentToSearch");
		return;
	}
	S8 category;
	std::string category_string = mPlacesCategory->getSelectedValue();
	if (category_string == "any")
	{
		category = LLParcel::C_ANY;
	}
	else
	{
		category = LLParcel::getCategoryFromString(category_string);
	}
	U32 scope = 0;
	if (gAgent.wantsPGOnly())
	{
		scope |= DFQ_PG_SIMS_ONLY;
	}
	bool adult_enabled = gAgent.canAccessAdult();
	bool mature_enabled = gAgent.canAccessMature();
	if (inc_pg)
	{
		scope |= DFQ_INC_PG;
	}
	if (inc_mature && mature_enabled)
	{
		scope |= DFQ_INC_MATURE;
	}
	if (inc_adult && adult_enabled)
	{
		scope |= DFQ_INC_ADULT;
	}
	scope |= DFQ_DWELL_SORT;

	mResultsReceived = 0;
	if (mQueryID.notNull())
	{
		mQueryID.setNull();
	}
	mQueryID.generate();

	if (mStartSearch < 0)
	{
		mStartSearch = 0;
	}

	gMessageSystem->newMessage("DirPlacesQuery");
	gMessageSystem->nextBlock("AgentData");
	gMessageSystem->addUUID("AgentID", gAgentID);
	gMessageSystem->addUUID("SessionID", gAgentSessionID);
	gMessageSystem->nextBlock("QueryData");
	gMessageSystem->addUUID("QueryID", getQueryID());
	gMessageSystem->addString("QueryText", text);
	gMessageSystem->addU32("QueryFlags", scope);
	gMessageSystem->addS8("Category", category);
	// TODO: Search filter by region name.
	gMessageSystem->addString("SimName", "");
	gMessageSystem->addS32("QueryStart", mStartSearch);
	gAgent.sendReliableMessage();
	LL_DEBUGS("Search") << "Firing off places search request: " << getQueryID() << LL_ENDL;

	mSearchResults->deleteAllItems();
	mSearchResults->setCommentText(LLTrans::getString("searching"));
	mNumResultsReturned = 0;
}

void FSPanelSearchPlaces::onBtnFind()
{
	std::string text = mSearchComboBox->getSimple();
	if (!text.empty())
	{
		LLSearchHistory::getInstance()->addEntry(text);
	}

	resetSearch();

	find();
}

void FSPanelSearchPlaces::onBtnNext()
{
	mStartSearch += RESULT_PAGE_SIZE;
	getChildView("places_back")->setEnabled(TRUE);

	find();
}

void FSPanelSearchPlaces::onBtnBack()
{
	mStartSearch -= RESULT_PAGE_SIZE;
	getChildView("places_back")->setEnabled(mStartSearch > 0);

	find();
}

void FSPanelSearchPlaces::resetSearch()
{
	mStartSearch = 0;
	getChildView("places_back")->setEnabled(FALSE);
	getChildView("places_next")->setEnabled(FALSE);
}

S32 FSPanelSearchPlaces::showNextButton(S32 rows)
{
	bool show_next_button = (mResultsReceived > RESULT_PAGE_SIZE);
	getChildView("places_next")->setEnabled(show_next_button);
	if (show_next_button)
	{
		rows -= (mResultsReceived - RESULT_PAGE_SIZE);
	}
	return rows;
}

void FSPanelSearchPlaces::onSelectItem()
{
	if (!mSearchResults)
	{
		return;
	}
	FSFloaterSearch* search_instance = LLFloaterReg::findTypedInstance<FSFloaterSearch>("search");
	if (search_instance)
	{
		search_instance->FSFloaterSearch::onSelectedItem(mSearchResults->getSelectedValue(), FSFloaterSearch::SC_PLACE);
	}
}

// static
void FSPanelSearchPlaces::processSearchReply(LLMessageSystem* msg, void**)
{
	LLUUID		agent_id;
	LLUUID		query_id;
	LLUUID		parcel_id;
	std::string	name;
	BOOL		for_sale;
	BOOL		auction;
	F32			dwell;

	msg->getUUID("AgentData", "AgentID", agent_id);
	msg->getUUID("QueryData", "QueryID", query_id);

	// Not for us
	if (agent_id != gAgentID)
	{
		return;
	}
	LL_DEBUGS("Search") << "received directory request - QueryID: " << query_id << " AgentID: " << agent_id << LL_ENDL;

	FSPanelSearchPlaces* self = FSFloaterSearch::getSearchPanel<FSPanelSearchPlaces>("panel_ls_places");

	// floater is closed or these are not results from our last request
	if (!self || query_id != self->getQueryID())
	{
		return;
	}

	LLScrollListCtrl* search_results = self->getChild<LLScrollListCtrl>("search_results_places");

	// Clear "Searching" label on first results
	if (self->mNumResultsReturned++ == 0)
	{
		search_results->deleteAllItems();
	}

	// Check for status messages
	if (msg->getNumberOfBlocks("StatusData"))
	{
		U32 status;
		msg->getU32("StatusData", "Status", status);
		if (status & STATUS_SEARCH_PLACES_FOUNDNONE)
		{
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("places_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
			return;
		}
		else if(status & STATUS_SEARCH_PLACES_SHORTSTRING)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_short"));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_BANNEDWORD)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_banned"));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_SEARCHDISABLED)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_disabled"));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_ESTATEEMPTY)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_disabled"));
			return;
		}
	}

	bool found_one = false;
	S32 num_new_rows = msg->getNumberOfBlocks("QueryReplies");
	if (num_new_rows == 0 && self->mResultsReceived == 0)
	{
		LLStringUtil::format_map_t map;
		map["[TEXT]"] = self->getChild<LLUICtrl>("places_edit")->getValue().asString();
		search_results->setEnabled(FALSE);
		search_results->setCommentText(LLTrans::getString("not_found", map));
	}

	self->mResultsReceived += num_new_rows;
	num_new_rows = self->showNextButton(num_new_rows);

	for (S32 i = 0; i < num_new_rows; i++)
	{
		msg->getUUID(	"QueryReplies",	"ParcelID",	parcel_id,	i);
		msg->getString(	"QueryReplies",	"Name",		name,		i);
		msg->getBOOL(	"QueryReplies",	"ForSale",	for_sale,i);
		msg->getBOOL(	"QueryReplies",	"Auction",	auction,	i);
		msg->getF32(	"QueryReplies",	"Dwell",	dwell,		i);
		if (parcel_id.isNull())
		{
			LL_DEBUGS("Search") << "Null result returned for QueryID: " << query_id << LL_ENDL;
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("places_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
		}
		else
		{
			LL_DEBUGS("Search") << "Got: " << name << " ParcelID: " << parcel_id << LL_ENDL;
			search_results->setEnabled(TRUE);
			found_one = true;

			LLSD content;
			LLSD element;

			element["id"] = parcel_id;

			if (auction)
			{
				element["columns"][0]["column"]	= "icon";
				element["columns"][0]["type"]	= "icon";
				element["columns"][0]["value"]	= "Icon_Auction";
			}
			else if (for_sale)
			{
				element["columns"][0]["column"]	= "icon";
				element["columns"][0]["type"]	= "icon";
				element["columns"][0]["value"]	= "Icon_For_Sale";
			}
			else
			{
				element["columns"][0]["column"]	= "icon";
				element["columns"][0]["type"]	= "icon";
				element["columns"][0]["value"]	= "Icon_Place";
			}

			element["columns"][1]["column"]	= "place_name";
			element["columns"][1]["value"]	= name;

			content["name"] = name;

			std::string buffer = llformat("%.0f", (F64)dwell);
			element["columns"][2]["column"]	= "dwell";
			element["columns"][2]["value"]	= buffer;

			search_results->addElement(element, ADD_BOTTOM);
			self->mResultsContent[parcel_id.asString()] = content;
		}
	}
	if (found_one)
	{
		search_results->selectFirstItem();
		search_results->setFocus(TRUE);
		self->onSelectItem();
	}
}

////////////////////////////////////////
//          Land Search Panel         //
////////////////////////////////////////

static LLPanelInjector<FSPanelSearchLand> t_panel_fs_search_land("panel_ls_land");

FSPanelSearchLand::FSPanelSearchLand() : FSSearchPanelBase()
, mQueryID(nullptr)
, mStartSearch(0)
, mResultsReceived(0)
, mResultsContent()
{
	mCommitCallbackRegistrar.add("CommitSearch", boost::bind(&FSPanelSearchLand::find, this));
}

FSPanelSearchLand::~FSPanelSearchLand()
{
}

BOOL FSPanelSearchLand::postBuild()
{
	mSearchResults	= getChild<LLScrollListCtrl>("search_results_land");
	mPriceEditor	= findChild<LLLineEditor>("price_edit");
	mAreaEditor		= findChild<LLLineEditor>("area_edit");
	if (mSearchResults)
	{
		mSearchResults->setCommitCallback(boost::bind(&FSPanelSearchLand::onSelectItem, this));
		mSearchResults->setEnabled(FALSE);
		mSearchResults->setCommentText(LLTrans::getString("no_results"));
	}
	if (mPriceEditor)
	{
		mPriceEditor->setCommitOnFocusLost(false);
		mPriceEditor->setCommitCallback(boost::bind(&FSPanelSearchLand::onBtnFind, this));
	}
	if (mAreaEditor)
	{
		mAreaEditor->setCommitOnFocusLost(false);
		mAreaEditor->setCommitCallback(boost::bind(&FSPanelSearchLand::find, this));
	}
	childSetAction("land_find", boost::bind(&FSPanelSearchLand::onBtnFind, this));
	childSetAction("land_next", boost::bind(&FSPanelSearchLand::onBtnNext, this));
	childSetAction("land_back", boost::bind(&FSPanelSearchLand::onBtnBack, this));

	getChildView("land_next")->setEnabled(FALSE);
	getChildView("land_back")->setEnabled(FALSE);
	
	return TRUE;
}

void FSPanelSearchLand::find()
{
	static LLUICachedControl<bool> inc_pg("ShowPGLand", 1);
	static LLUICachedControl<bool> inc_mature("ShowMatureLand", 0);
	static LLUICachedControl<bool> inc_adult("ShowAdultLand", 0);
	static LLUICachedControl<bool> limit_price("FindLandPrice", 1);
	static LLUICachedControl<bool> limit_area("FindLandArea", 1);
	if (!(inc_pg || inc_mature || inc_adult))
	{
		LLNotificationsUtil::add("NoContentToSearch");
		return;
	}

	U32 category = ST_ALL;
	const std::string& selection = findChild<LLComboBox>("land_category")->getSelectedValue().asString();
	if (!selection.empty())
	{
		if (selection == "Auction")
		{
			category = ST_AUCTION;
		}
		else if (selection == "Mainland")
		{
			category = ST_MAINLAND;
		}
		else if (selection == "Estate")
		{
			category = ST_ESTATE;
		}
	}

	U32 scope = 0;
	if (gAgent.wantsPGOnly())
	{
		scope |= DFQ_PG_SIMS_ONLY;
	}
	bool mature_enabled = gAgent.canAccessMature();
	bool adult_enabled = gAgent.canAccessAdult();
	if (inc_pg)
	{
		scope |= DFQ_INC_PG;
	}
	if (inc_mature && mature_enabled)
	{
		scope |= DFQ_INC_MATURE;
	}
	if (inc_adult && adult_enabled)
	{
		scope |= DFQ_INC_ADULT;
	}
	const std::string& sort = findChild<LLComboBox>("land_sort_combo")->getSelectedValue().asString();
	if (!sort.empty())
	{
		if (sort == "Name")
		{
			scope |= DFQ_NAME_SORT;
		}
		else if (sort == "Price")
		{
			scope |= DFQ_PRICE_SORT;
		}
		else if (sort == "PPM")
		{
			scope |= DFQ_PER_METER_SORT;
		}
		else if (sort == "Area")
		{
			scope |= DFQ_AREA_SORT;
		}
	}
	else
	{
		scope |= DFQ_PRICE_SORT;
	}
	if (childGetValue("ascending_check").asBoolean())
	{
		scope |= DFQ_SORT_ASC;
	}
	if (limit_price)
	{
		scope |= DFQ_LIMIT_BY_PRICE;
	}
	if (limit_area)
	{
		scope |= DFQ_LIMIT_BY_AREA;
	}
	S32 price = childGetValue("edit_price").asInteger();
	S32 area = childGetValue("edit_area").asInteger();

	mResultsReceived = 0;
	if (mQueryID.notNull())
	{
		mQueryID.setNull();
	}
	mQueryID.generate();

	if (mStartSearch < 0)
	{
		mStartSearch = 0;
	}

	gMessageSystem->newMessage("DirLandQuery");
	gMessageSystem->nextBlock("AgentData");
	gMessageSystem->addUUID("AgentID", gAgentID);
	gMessageSystem->addUUID("SessionID", gAgentSessionID);
	gMessageSystem->nextBlock("QueryData");
	gMessageSystem->addUUID("QueryID", getQueryID());
	gMessageSystem->addU32("QueryFlags", scope);
	gMessageSystem->addU32("SearchType", category);
	gMessageSystem->addS32("Price", price);
	gMessageSystem->addS32("Area", area);
	gMessageSystem->addS32("QueryStart", mStartSearch);
	gAgent.sendReliableMessage();
	LL_DEBUGS("Search") << "Firing off places search request: " << getQueryID() << category << LL_ENDL;

	mSearchResults->deleteAllItems();
	mSearchResults->setCommentText(LLTrans::getString("searching"));
	mNumResultsReturned = 0;
}

void FSPanelSearchLand::onBtnFind()
{
	resetSearch();

	find();
}

void FSPanelSearchLand::onBtnNext()
{
	mStartSearch += RESULT_PAGE_SIZE;
	getChildView("land_back")->setEnabled(TRUE);

	find();
}

void FSPanelSearchLand::onBtnBack()
{
	mStartSearch -= RESULT_PAGE_SIZE;
	getChildView("land_back")->setEnabled(mStartSearch > 0);

	find();
}

void FSPanelSearchLand::resetSearch()
{
	mStartSearch = 0;
	getChildView("land_back")->setEnabled(FALSE);
	getChildView("land_next")->setEnabled(FALSE);
}

S32 FSPanelSearchLand::showNextButton(S32 rows)
{
	bool show_next_button = (mResultsReceived > RESULT_PAGE_SIZE);
	getChildView("land_next")->setEnabled(show_next_button);
	if (show_next_button)
	{
		rows -= (mResultsReceived - RESULT_PAGE_SIZE);
	}
	return rows;
}

void FSPanelSearchLand::onSelectItem()
{
	if (!mSearchResults)
	{
		return;
	}
	FSFloaterSearch* search_instance = LLFloaterReg::findTypedInstance<FSFloaterSearch>("search");
	if (search_instance)
	{
		search_instance->FSFloaterSearch::onSelectedItem(mSearchResults->getSelectedValue(), FSFloaterSearch::SC_PLACE);
	}
}

// static
void FSPanelSearchLand::processSearchReply(LLMessageSystem* msg, void**)
{
	LLUUID		agent_id;
	LLUUID		query_id;
	LLUUID		parcel_id;
	std::string	name;
	std::string	land_sku;
	std::string	land_type;
	BOOL		auction;
	BOOL		for_sale;
	S32			price;
	S32			area;

	msg->getUUID("AgentData", "AgentID", agent_id);
	msg->getUUID("QueryData", "QueryID", query_id);

	// Not for us
	if (agent_id != gAgentID)
	{
		return;
	}
	LL_DEBUGS("Search") << "received directory request - QueryID: " << query_id << " AgentID: " << agent_id << LL_ENDL;

	FSPanelSearchLand* self = FSFloaterSearch::getSearchPanel<FSPanelSearchLand>("panel_ls_land");

	// floater is closed or these are not results from our last request
	if (!self || query_id != self->mQueryID)
	{
		return;
	}

	LLScrollListCtrl* search_results = self->getChild<LLScrollListCtrl>("search_results_land");
	// clear "Searching" label on first results
	if (self->mNumResultsReturned++ == 0)
	{
		search_results->deleteAllItems();
	}

	static LLUICachedControl<bool> use_price("FindLandPrice", 1);
	static LLUICachedControl<bool> use_area("FindLandArea", 1);
	S32 limit_price = self->childGetValue("edit_price").asInteger();
	S32 limit_area = self->childGetValue("edit_area").asInteger();

	bool found_one = false;
	S32 num_new_rows = msg->getNumberOfBlocks("QueryReplies");
	if (num_new_rows == 0 && self->mResultsReceived == 0)
	{
		LLStringUtil::format_map_t map;
		map["[TEXT]"] = self->getChild<LLUICtrl>("events_edit")->getValue().asString();
		search_results->setEnabled(FALSE);
		search_results->setCommentText(LLTrans::getString("not_found", map));
	}
	self->mResultsReceived += num_new_rows;

	S32 not_auction = 0;
	for (S32 i = 0; i < num_new_rows; i++)
	{
		msg->getUUID(	"QueryReplies", "ParcelID",		parcel_id,	i);
		msg->getString(	"QueryReplies", "Name",			name,		i);
		msg->getBOOL(	"QueryReplies", "Auction",		auction,	i);
		msg->getBOOL(	"QueryReplies", "ForSale",		for_sale,	i);
		msg->getS32(	"QueryReplies", "SalePrice",	price,		i);
		msg->getS32(	"QueryReplies", "ActualArea",	area,		i);
		if (parcel_id.isNull())
		{
			LL_DEBUGS("Search") << "Null result returned for QueryID: " << query_id << LL_ENDL;
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("no_results"));
		}
		else
		{
			LL_DEBUGS("Search") << "Got: " << name << " ClassifiedID: " << parcel_id << LL_ENDL;
			search_results->setEnabled(TRUE);
			found_one = true;
			if (msg->getSizeFast(_PREHASH_QueryReplies, i, _PREHASH_ProductSKU) > 0)
			{
				msg->getStringFast(	_PREHASH_QueryReplies, _PREHASH_ProductSKU, land_sku, i);
				land_type = LLProductInfoRequestManager::instance().getDescriptionForSku(land_sku);
			}
			else
			{
				land_sku.clear();
				land_type = LLTrans::getString("land_type_unknown");
			}
			if (parcel_id.isNull())
			{
				continue;
			}
			if (use_price && (price > limit_price))
			{
				continue;
			}
			if (use_area && (area < limit_area))
			{
				continue;
			}

			LLSD content;
			LLSD element;

			element["id"] = parcel_id;
			if (auction)
			{
				element["columns"][0]["column"]	= "icon";
				element["columns"][0]["type"]	= "icon";
				element["columns"][0]["value"]	= "Icon_Auction";
			}
			else if (for_sale)
			{
				element["columns"][0]["column"]	= "icon";
				element["columns"][0]["type"]	= "icon";
				element["columns"][0]["value"]	= "Icon_For_Sale";
			}
			else
			{
				element["columns"][0]["column"]	= "icon";
				element["columns"][0]["type"]	= "icon";
				element["columns"][0]["value"]	= "Icon_Place";
			}

			element["columns"][1]["column"]	= "land_name";
			element["columns"][1]["value"]	= name;

			content["place_name"] = name;

			std::string buffer = "Auction";
			if (!auction)
			{
				buffer = llformat("%d", price);
				not_auction++;
			}
			element["columns"][2]["column"]	= "price";
			element["columns"][2]["value"]	= price;

			element["columns"][3]["column"]	= "area";
			element["columns"][3]["value"]	= area;
			if (!auction)
			{
				F32 ppm;
				if (area > 0)
				{
					ppm = (F32)price / (F32)area;
				}
				else
				{
					ppm = 0.f;
				}
				std::string ppm_buffer = llformat("%.1f", ppm);
				element["columns"][4]["column"]	= "ppm";
				element["columns"][4]["value"] = ppm_buffer;
			}
			else
			{
				element["columns"][4]["column"]	= "ppm";
				element["columns"][4]["value"]	= "1.0";
			}

			element["columns"][5]["column"]	= "land_type";
			element["columns"][5]["value"]	= land_type;

			search_results->addElement(element, ADD_BOTTOM);
			self->mResultsContent[parcel_id.asString()] = content;
		}
		// We test against non-auction properties because they don't count towards the page limit.
		self->showNextButton(not_auction);
	}
	if (found_one)
	{
		search_results->selectFirstItem();
		search_results->setFocus(TRUE);
		self->onSelectItem();
	}
}

////////////////////////////////////////
//      Classifieds Search Panel      //
////////////////////////////////////////

static LLPanelInjector<FSPanelSearchClassifieds> t_panel_fs_search_classifieds("panel_ls_classifieds");

FSPanelSearchClassifieds::FSPanelSearchClassifieds() : FSSearchPanelBase()
, mQueryID(nullptr)
, mStartSearch(0)
, mResultsReceived(0)
, mResultsContent()
{
	mCommitCallbackRegistrar.add("CommitSearch", boost::bind(&FSPanelSearchClassifieds::find, this));
}

FSPanelSearchClassifieds::~FSPanelSearchClassifieds()
{
}

BOOL FSPanelSearchClassifieds::postBuild()
{
	mSearchComboBox = findChild<LLSearchComboBox>("classifieds_edit");
	mSearchResults = getChild<LLScrollListCtrl>("search_results_classifieds");
	if (mSearchComboBox)
	{
		mSearchComboBox->setCommitCallback(boost::bind(&FSPanelSearchClassifieds::onBtnFind, this));
		fillSearchComboBox(mSearchComboBox);
	}
	if (mSearchResults)
	{
		mSearchResults->setCommitCallback(boost::bind(&FSPanelSearchClassifieds::onSelectItem, this));
		mSearchResults->setEnabled(FALSE);
		mSearchResults->setCommentText(LLTrans::getString("no_results"));
	}

	mClassifiedsCategory = getChild<LLComboBox>("classifieds_category");
	if (mClassifiedsCategory)
	{
		LLClassifiedInfo::cat_map::iterator iter;
		mClassifiedsCategory->add(LLTrans::getString("all_categories"), LLSD(0));
		mClassifiedsCategory->addSeparator();
		for (iter = LLClassifiedInfo::sCategories.begin();
			 iter != LLClassifiedInfo::sCategories.end();
			 iter++)
		{
			mClassifiedsCategory->add(LLTrans::getString(iter->second), LLSD((S32)iter->first));
		}
	}
	childSetAction("classifieds_next", boost::bind(&FSPanelSearchClassifieds::onBtnNext, this));
	childSetAction("classifieds_back", boost::bind(&FSPanelSearchClassifieds::onBtnBack, this));

	getChildView("classifieds_next")->setEnabled(FALSE);
	getChildView("classifieds_back")->setEnabled(FALSE);
	
	return TRUE;
}

void FSPanelSearchClassifieds::focusDefaultElement()
{
	mSearchComboBox->focusTextEntry();
}

void FSPanelSearchClassifieds::find()
{
	std::string text = filterShortWords(mSearchComboBox->getSimple());
	if (text.size() == 0)
	{
		mSearchResults->setCommentText(LLTrans::getString("search_short"));
		return;
	}

	static LLUICachedControl<bool> inc_pg("ShowPGClassifieds", 1);
	static LLUICachedControl<bool> inc_mature("ShowMatureClassifieds", 0);
	static LLUICachedControl<bool> inc_adult("ShowAdultClassifieds", 0);
	if (!(inc_pg || inc_mature || inc_adult))
	{
		LLNotificationsUtil::add("NoContentToSearch");
		return;
	}
	U32 category = mClassifiedsCategory->getValue().asInteger();
	BOOL auto_renew = FALSE;
	U32 flags = pack_classified_flags_request(auto_renew, inc_pg, inc_mature, inc_adult);

	mResultsReceived = 0;
	if (mQueryID.notNull())
	{
		mQueryID.setNull();
	}
	mQueryID.generate();

	if (mStartSearch < 0)
	{
		mStartSearch = 0;
	}

	gMessageSystem->newMessageFast(_PREHASH_DirClassifiedQuery);
	gMessageSystem->nextBlockFast(_PREHASH_AgentData);
	gMessageSystem->addUUIDFast(_PREHASH_AgentID, gAgentID);
	gMessageSystem->addUUIDFast(_PREHASH_SessionID, gAgentSessionID);
	gMessageSystem->nextBlockFast(_PREHASH_QueryData);
	gMessageSystem->addUUIDFast(_PREHASH_QueryID, getQueryID());
	gMessageSystem->addStringFast(_PREHASH_QueryText, text);
	gMessageSystem->addU32Fast(_PREHASH_QueryFlags, flags);
	gMessageSystem->addU32Fast(_PREHASH_Category, category);
	gMessageSystem->addS32Fast(_PREHASH_QueryStart, mStartSearch);
	gAgent.sendReliableMessage();
	LL_DEBUGS("Search") << "Firing off classified ad search request: " << getQueryID() << LL_ENDL;

	mSearchResults->deleteAllItems();
	mSearchResults->setCommentText(LLTrans::getString("searching"));
	mNumResultsReturned = 0;
}

void FSPanelSearchClassifieds::onBtnFind()
{
	std::string text = mSearchComboBox->getSimple();
	if (!text.empty())
	{
		LLSearchHistory::getInstance()->addEntry(text);
	}

	resetSearch();

	find();
}

void FSPanelSearchClassifieds::onBtnNext()
{
	mStartSearch += RESULT_PAGE_SIZE;
	getChildView("classifieds_back")->setEnabled(TRUE);

	find();
}

void FSPanelSearchClassifieds::onBtnBack()
{
	mStartSearch -= RESULT_PAGE_SIZE;
	getChildView("classifieds_back")->setEnabled(mStartSearch > 0);

	find();
}

void FSPanelSearchClassifieds::resetSearch()
{
	mStartSearch = 0;
	getChildView("classifieds_back")->setEnabled(FALSE);
	getChildView("classifieds_next")->setEnabled(FALSE);
}

S32 FSPanelSearchClassifieds::showNextButton(S32 rows)
{
	bool show_next_button = (mResultsReceived > RESULT_PAGE_SIZE);
	getChildView("classifieds_next")->setEnabled(show_next_button);
	if (show_next_button)
	{
		rows -= (mResultsReceived - RESULT_PAGE_SIZE);
	}
	return rows;
}

void FSPanelSearchClassifieds::onSelectItem()
{
	if (!mSearchResults)
	{
		return;
	}
	FSFloaterSearch* search_instance = LLFloaterReg::findTypedInstance<FSFloaterSearch>("search");
	if (search_instance)
	{
		search_instance->FSFloaterSearch::onSelectedItem(mSearchResults->getSelectedValue(), FSFloaterSearch::SC_CLASSIFIED);
	}
}

// static
void FSPanelSearchClassifieds::processSearchReply(LLMessageSystem* msg, void**)
{
	LLUUID		agent_id;
	LLUUID		query_id;
	LLUUID		classified_id;
	std::string	name;
	U32			creation_date;
	U32			expiration_date;
	S32			price_for_listing;

	msg->getUUID("AgentData", "AgentID", agent_id);
	msg->getUUID("QueryData", "QueryID", query_id);

	// Not for us
	if (agent_id != gAgentID)
	{
		return;
	}
	LL_DEBUGS("Search") << "received directory request - QueryID: " << query_id << " AgentID: " << agent_id << LL_ENDL;

	FSPanelSearchClassifieds* self = FSFloaterSearch::getSearchPanel<FSPanelSearchClassifieds>("panel_ls_classifieds");

	if (msg->getNumberOfBlocks("StatusData"))
	{
		U32 status;
		msg->getU32("StatusData", "Status", status);
		if (status & STATUS_SEARCH_CLASSIFIEDS_BANNEDWORD)
		{
			LLNotificationsUtil::add("SearchWordBanned");
		}
	}

	// floater is closed or these are not results from our last request
	if (!self || query_id != self->mQueryID)
	{
		return;
	}

	LLScrollListCtrl* search_results = self->getChild<LLScrollListCtrl>("search_results_classifieds");

	// Clear "Searching" label on first results
	if (self->mNumResultsReturned++ == 0)
	{
		search_results->deleteAllItems();
	}

	// Check for status messages
	if (msg->getNumberOfBlocks("StatusData"))
	{
		U32 status;
		msg->getU32("StatusData", "Status", status);
		if (status & STATUS_SEARCH_PLACES_FOUNDNONE)
		{
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("classifieds_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
			return;
		}
		else if(status & STATUS_SEARCH_PLACES_SHORTSTRING)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_short"));
			return;
		}
		else if (status & STATUS_SEARCH_CLASSIFIEDS_BANNEDWORD)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_banned"));
			return;
		}
		else if (status & STATUS_SEARCH_PLACES_SEARCHDISABLED)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_disabled"));
			return;
		}
	}

	bool found_one = false;
	S32 num_new_rows = msg->getNumberOfBlocks("QueryReplies");
	if (num_new_rows == 0 && self->mResultsReceived == 0)
	{
		LLStringUtil::format_map_t map;
		map["[TEXT]"] = self->getChild<LLUICtrl>("classifieds_edit")->getValue().asString();
		search_results->setEnabled(FALSE);
		search_results->setCommentText(LLTrans::getString("not_found", map));
	}
	self->mResultsReceived += num_new_rows;
	num_new_rows = self->showNextButton(num_new_rows);

	for (S32 i = 0; i < num_new_rows; i++)
	{
		msg->getUUID(	"QueryReplies", "ClassifiedID",		classified_id,	i);
		msg->getString(	"QueryReplies", "Name",				name,			i);
		msg->getU32(	"QueryReplies", "CreationDate",		creation_date,	i);
		msg->getU32(	"QueryReplies", "ExpirationDate",	expiration_date,i);
		msg->getS32(	"QueryReplies", "PriceForListing",	price_for_listing,i);
		if (classified_id.isNull())
		{
			LL_DEBUGS("Search") << "Null result returned for QueryID: " << query_id << LL_ENDL;
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("classifieds_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
		}
		else
		{
			LL_DEBUGS("Search") << "Got: " << name << " ClassifiedID: " << classified_id << LL_ENDL;
			search_results->setEnabled(TRUE);
			found_one = true;

			LLSD content;
			LLSD element;

			element["id"] = classified_id;

			element["columns"][0]["column"]	= "icon";
			element["columns"][0]["type"]	= "icon";
			element["columns"][0]["value"]	= "icon_top_pick.tga";

			element["columns"][1]["column"]	= "classified_name";
			element["columns"][1]["value"]	= name;

			element["columns"][2]["column"]	= "price";
			element["columns"][2]["value"]	= price_for_listing;

			content["name"] = name;

			search_results->addElement(element, ADD_BOTTOM);
			self->mResultsContent[classified_id.asString()] = content;
		}
	}
	if (found_one)
	{
		search_results->selectFirstItem();
		search_results->setFocus(TRUE);
		self->onSelectItem();
	}
}

////////////////////////////////////////
//        Events Search Panel         //
////////////////////////////////////////

static LLPanelInjector<FSPanelSearchEvents> t_panel_fs_search_events("panel_ls_events");

FSPanelSearchEvents::FSPanelSearchEvents() : FSSearchPanelBase()
, mQueryID(nullptr)
, mResultsReceived(0)
, mStartSearch(0)
, mDay(0)
, mResultsContent()
{
	mCommitCallbackRegistrar.add("CommitSearch", boost::bind(&FSPanelSearchEvents::find, this));
}

FSPanelSearchEvents::~FSPanelSearchEvents()
{
}

BOOL FSPanelSearchEvents::postBuild()
{
	mSearchComboBox = findChild<LLSearchComboBox>("events_edit");
	mSearchResults = getChild<LLScrollListCtrl>("search_results_events");
	mEventsMode = findChild<LLRadioGroup>("events_search_mode");
	if (mSearchComboBox)
	{
		mSearchComboBox->setCommitCallback(boost::bind(&FSPanelSearchEvents::onBtnFind, this));
		fillSearchComboBox(mSearchComboBox);
	}
	if (mSearchResults)
	{
		mSearchResults->setCommitCallback(boost::bind(&FSPanelSearchEvents::onSelectItem, this));
		mSearchResults->setEnabled(FALSE);
		mSearchResults->setCommentText(LLTrans::getString("no_results"));
	}
	if (mEventsMode)
	{
		mEventsMode->setCommitCallback(boost::bind(&FSPanelSearchEvents::onSearchModeChanged, this));
		mEventsMode->selectFirstItem();
	}

	childSetAction("events_next", boost::bind(&FSPanelSearchEvents::onBtnNext, this));
	childSetAction("events_back", boost::bind(&FSPanelSearchEvents::onBtnBack, this));
	childSetAction("events_tomorrow", boost::bind(&FSPanelSearchEvents::onBtnTomorrow, this));
	childSetAction("events_yesterday", boost::bind(&FSPanelSearchEvents::onBtnYesterday, this));
	childSetAction("events_today", boost::bind(&FSPanelSearchEvents::onBtnToday, this));

	getChildView("events_next")->setEnabled(FALSE);
	getChildView("events_back")->setEnabled(FALSE);
	getChildView("events_tomorrow")->setEnabled(FALSE);
	getChildView("events_yesterday")->setEnabled(FALSE);
	getChildView("events_today")->setEnabled(FALSE);
	setDay(0);

	return TRUE;
}

void FSPanelSearchEvents::focusDefaultElement()
{
	mSearchComboBox->focusTextEntry();
}

void FSPanelSearchEvents::find()
{
	std::string text = filterShortWords(mSearchComboBox->getSimple());

	static LLUICachedControl<bool> inc_pg("ShowPGEvents", 1);
	static LLUICachedControl<bool> inc_mature("ShowMatureEvents", 0);
	static LLUICachedControl<bool> inc_adult("ShowAdultEvents", 0);
	if (!(inc_pg || inc_mature || inc_adult))
	{
		LLNotificationsUtil::add("NoContentToSearch");
		return;
	}

	U32 category = findChild<LLComboBox>("events_category")->getSelectedValue().asInteger();
	U32 scope = DFQ_DATE_EVENTS;
	if (gAgent.wantsPGOnly())
	{
		scope |= DFQ_PG_SIMS_ONLY;
	}
	bool mature_enabled = gAgent.canAccessMature();
	bool adult_enabled = gAgent.canAccessAdult();
	if (inc_pg)
	{
		scope |= DFQ_INC_PG;
	}
	if (inc_mature && mature_enabled)
	{
		scope |= DFQ_INC_MATURE;
	}
	if (inc_adult && adult_enabled)
	{
		scope |= DFQ_INC_ADULT;
	}

	std::ostringstream string;

	if ("current" == childGetValue("events_search_mode").asString())
	{
		string << "u|";
	}
	else
	{
		string << mDay << "|";
	}
	string << category << "|";
	string << text;

	mResultsReceived = 0;
	if (mQueryID.notNull())
	{
		mQueryID.setNull();
	}
	mQueryID.generate();

	if (mStartSearch < 0)
	{
		mStartSearch = 0;
	}

	gMessageSystem->newMessage("DirFindQuery");
	gMessageSystem->nextBlock("AgentData");
	gMessageSystem->addUUID("AgentID", gAgentID);
	gMessageSystem->addUUID("SessionID", gAgentSessionID);
	gMessageSystem->nextBlock("QueryData");
	gMessageSystem->addUUID("QueryID", getQueryID());
	gMessageSystem->addString("QueryText", string.str());
	gMessageSystem->addU32("QueryFlags", scope);
	gMessageSystem->addS32("QueryStart", mStartSearch);
	gAgent.sendReliableMessage();
	LL_DEBUGS("Search") << "Firing off search request: " << getQueryID() << " Search Text: " << string.str() << LL_ENDL;

	mSearchResults->deleteAllItems();
	mSearchResults->setCommentText(LLTrans::getString("searching"));
	mNumResultsReturned = 0;
}

void FSPanelSearchEvents::onBtnFind()
{
	std::string text = mSearchComboBox->getSimple();
	if (!text.empty())
	{
		LLSearchHistory::getInstance()->addEntry(text);
	}

	resetSearch();

	find();
}

void FSPanelSearchEvents::onBtnNext()
{
	mStartSearch += RESULT_PAGE_SIZE;
	getChildView("events_back")->setEnabled(TRUE);

	find();
}

void FSPanelSearchEvents::onBtnBack()
{
	mStartSearch -= RESULT_PAGE_SIZE;
	getChildView("events_back")->setEnabled(mStartSearch > 0);

	find();
}

void FSPanelSearchEvents::onBtnTomorrow()
{
	resetSearch();
	setDay(mDay + 1);

	find();
}

void FSPanelSearchEvents::onBtnYesterday()
{
	resetSearch();
	setDay(mDay - 1);

	find();
}

void FSPanelSearchEvents::onBtnToday()
{
	resetSearch();
	setDay(0);

	find();
}

void FSPanelSearchEvents::resetSearch()
{
	mStartSearch = 0;
	getChildView("events_back")->setEnabled(FALSE);
	getChildView("events_next")->setEnabled(FALSE);
}

void FSPanelSearchEvents::onSearchModeChanged()
{
	if (mEventsMode->getValue().asString() == "current")
	{
		getChildView("events_yesterday")->setEnabled(FALSE);
		getChildView("events_tomorrow")->setEnabled(FALSE);
		getChildView("events_today")->setEnabled(FALSE);
	}
	else
	{
		getChildView("events_yesterday")->setEnabled(TRUE);
		getChildView("events_tomorrow")->setEnabled(TRUE);
		getChildView("events_today")->setEnabled(TRUE);
	}
}

void FSPanelSearchEvents::setDay(S32 day)
{
	mDay = day;
	struct tm* internal_time;

	time_t utc = time_corrected();
	utc += day * 24 * 60 * 60;
	internal_time = utc_to_pacific_time(utc, is_daylight_savings());
	std::string buffer = llformat("%d/%d", 1 + internal_time->tm_mon, internal_time->tm_mday);
	childSetValue("events_date", buffer);
}

S32 FSPanelSearchEvents::showNextButton(S32 rows)
{
	bool show_next_button = (mResultsReceived > RESULT_PAGE_SIZE);
	getChildView("events_next")->setEnabled(show_next_button);
	if (show_next_button)
	{
		rows -= (mResultsReceived - RESULT_PAGE_SIZE);
	}
	return rows;
}

void FSPanelSearchEvents::onSelectItem()
{
	if (!mSearchResults)
	{
		return;
	}
	S32 event_id = mSearchResults->getSelectedValue();
	FSFloaterSearch* search_instance = LLFloaterReg::findTypedInstance<FSFloaterSearch>("search");
	if (search_instance)
	{
		search_instance->FSFloaterSearch::onSelectedEvent(event_id);
	}
}

// static
void FSPanelSearchEvents::processSearchReply(LLMessageSystem* msg, void**)
{
	LLUUID		agent_id;
	LLUUID		query_id;
	LLUUID		owner_id;
	std::string	name;
	std::string	date;

	msg->getUUID("AgentData", "AgentID", agent_id);
	msg->getUUID("QueryData", "QueryID", query_id);

	// Not for us
	if (agent_id != gAgentID)
	{
		return;
	}
	LL_DEBUGS("Search") << "received directory request - QueryID: " << query_id << " AgentID: " << agent_id << LL_ENDL;

	FSPanelSearchEvents* self = FSFloaterSearch::getSearchPanel<FSPanelSearchEvents>("panel_ls_events");

	// floater is closed or these are not results from our last request
	if (!self || query_id != self->mQueryID)
	{
		return;
	}

	LLScrollListCtrl* search_results = self->getChild<LLScrollListCtrl>("search_results_events");

	// Clear "Searching" label on first results
	if (self->mNumResultsReturned++ == 0)
	{
		search_results->deleteAllItems();
	}
	// Check for status messages
	if (msg->getNumberOfBlocks("StatusData"))
	{
		U32 status;
		msg->getU32("StatusData", "Status", status);
		if (status & STATUS_SEARCH_EVENTS_FOUNDNONE)
		{
			LLStringUtil::format_map_t map;
			map["[TEXT]"] = self->getChild<LLUICtrl>("events_edit")->getValue().asString();
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("not_found", map));
			return;
		}
		else if(status & STATUS_SEARCH_EVENTS_SHORTSTRING)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_short"));
			return;
		}
		else if (status & STATUS_SEARCH_EVENTS_BANNEDWORD)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_banned"));
			return;
		}
		else if (status & STATUS_SEARCH_EVENTS_SEARCHDISABLED)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_disabled"));
			return;
		}
		else if (status & STATUS_SEARCH_EVENTS_NODATEOFFSET)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_no_date_offset"));
			return;
		}
		else if (status & STATUS_SEARCH_EVENTS_NOCATEGORY)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_no_events_category"));
			return;
		}
		else if (status & STATUS_SEARCH_EVENTS_NOQUERY)
		{
			search_results->setEnabled(FALSE);
			search_results->setCommentText(LLTrans::getString("search_no_query"));
			return;
		}
	}

	S32 num_new_rows = msg->getNumberOfBlocks("QueryReplies");
	if (num_new_rows == 0 && self->mResultsReceived == 0)
	{
		LLStringUtil::format_map_t map;
		map["[TEXT]"] = self->getChild<LLUICtrl>("events_edit")->getValue().asString();
		search_results->setEnabled(FALSE);
		search_results->setCommentText(LLTrans::getString("not_found", map));
	}

	self->mResultsReceived += num_new_rows;
	num_new_rows = self->showNextButton(num_new_rows);
	static LLUICachedControl<bool> inc_pg("ShowPGEvents", 1);
	static LLUICachedControl<bool> inc_mature("ShowMatureEvents", 0);
	static LLUICachedControl<bool> inc_adult("ShowAdultEvents", 0);
	bool found_one = false;

	for (S32 i = 0; i < num_new_rows; i++)
	{
		U32 event_id;
		U32 unix_time;
		U32 event_flags;

		msg->getUUID(	"QueryReplies",	"OwnerID",		owner_id,	i);
		msg->getString(	"QueryReplies",	"Name",			name,		i);
		msg->getU32(	"QueryReplies",	"EventID",		event_id,	i);
		msg->getString(	"QueryReplies",	"Date",			date,		i);
		msg->getU32(	"QueryReplies",	"UnixTime",		unix_time,	i);
		msg->getU32(	"QueryReplies",	"EventFlags",	event_flags,i);

		// Skip empty events...
		if (owner_id.isNull())
		{
			LL_INFOS("Search") << "Skipped " << event_id << " because of a nullptr owner result" << LL_ENDL;
			continue;
		}
		// Skips events that don't match our scope...
		if (((event_flags & (EVENT_FLAG_ADULT | EVENT_FLAG_MATURE)) == EVENT_FLAG_NONE) && !inc_pg)
		{
			LL_INFOS("Search") << "Skipped " << event_id << " because it was out of scope" << LL_ENDL;
			continue;
		}
		if ((event_flags & EVENT_FLAG_MATURE) && !inc_mature)
		{
			LL_INFOS("Search") << "Skipped " << event_id << " because it was out of scope" << LL_ENDL;
			continue;
		}
		if ((event_flags & EVENT_FLAG_ADULT) && !inc_adult)
		{
			LL_INFOS("Search") << "Skipped " << event_id << " because it was out of scope" << LL_ENDL;
			continue;
		}
		search_results->setEnabled(TRUE);
		found_one = true;

		LLSD content;
		LLSD element;

		element["id"] = llformat("%u", event_id);

		if (event_flags == EVENT_FLAG_ADULT)
		{
			element["columns"][0]["column"] = "icon";
			element["columns"][0]["type"] = "icon";
			element["columns"][0]["value"] = "Icon_Legacy_Event_Adult";
		}
		else if (event_flags == EVENT_FLAG_MATURE)
		{
			element["columns"][0]["column"] = "icon";
			element["columns"][0]["type"] = "icon";
			element["columns"][0]["value"] = "Icon_Legacy_Event_Mature";
		}
		else
		{
			element["columns"][0]["column"] = "icon";
			element["columns"][0]["type"] = "icon";
			element["columns"][0]["value"] = "Icon_Legacy_Event_PG";
		}
		element["columns"][1]["column"] = "name";
		element["columns"][1]["value"] = name;

		element["columns"][2]["column"] = "date";
		element["columns"][2]["value"] = date;

		element["columns"][3]["column"] = "time";
		element["columns"][3]["value"] = llformat("%u", unix_time);

		content["name"] = name;
		content["event_id"] = (S32)event_id;

		search_results->addElement(element, ADD_BOTTOM);
		std::string event = llformat("%u", event_id);
		self->mResultsContent[event] = content;
	}
	if (found_one)
	{
		search_results->selectFirstItem();
		search_results->setFocus(TRUE);
		self->onSelectItem();
	}
}

////////////////////////////////////////
//          WebSearch Panel           //
////////////////////////////////////////

static LLPanelInjector<FSPanelSearchWeb> t_panel_fs_search_web("panel_ls_web");

FSPanelSearchWeb::FSPanelSearchWeb() : FSSearchPanelBase()
, mWebBrowser(nullptr)
, mResetFocusOnLoad(false)
{
	// Second Life grids use a different URL format now
	mCategoryPaths = LLSD::emptyMap();
	if (LLGridManager::getInstance()->isInSecondLife())
	{
		// declare a map that transforms a category name into
		// the parameter list that is used to search that category
		mCategoryPaths["people"]       = "collection_chosen=people";
		mCategoryPaths["places"]       = "collection_chosen=places";
		mCategoryPaths["events"]       = "collection_chosen=events";
		mCategoryPaths["groups"]       = "collection_chosen=groups";
		mCategoryPaths["destinations"] = "collection_chosen=destinations";

		mCategoryPaths["classifieds"]  = "search_type=classified";
		mCategoryPaths["wiki"]         = "search/wiki";						// not sure if this is still a thing in the new search

		mCategoryPaths["all"]          = mCategoryPaths["people"].asString() + "&" +
										mCategoryPaths["places"].asString() + "&" +
										mCategoryPaths["events"].asString() + "&" +
										mCategoryPaths["groups"].asString() + "&" +
										mCategoryPaths["destinations"].asString();
	}
	// OpenSim currently still uses the old URL format
	else
	{
		// declare a map that transforms a category name into
		// the URL suffix that is used to search that category
		mCategoryPaths["all"]          = "search";
		mCategoryPaths["people"]       = "search/people";
		mCategoryPaths["places"]       = "search/places";
		mCategoryPaths["events"]       = "search/events";
		mCategoryPaths["groups"]       = "search/groups";
		mCategoryPaths["wiki"]         = "search/wiki";
		mCategoryPaths["destinations"] = "destinations";
		mCategoryPaths["classifieds"]  = "classifieds";
	}
}

BOOL FSPanelSearchWeb::postBuild()
{
	mWebBrowser = getChild<LLMediaCtrl>("search_browser");
	return TRUE;
}

void FSPanelSearchWeb::loadURL(const SearchQuery &p)
{
	if (!mWebBrowser || !p.validateBlock())
	{
		return;
	}

	// CATEGORY is no longer used as part of the path on Second Life grids
	LLSD subs = LLSD().with("CATEGORY", "");

	// on OpenSim grids it probably is currently still being used, so keep the old behavior
	if (!LLGridManager::getInstance()->isInSecondLife())
	{
		// work out the subdir to use based on the requested category
		LLSD subs = LLSD().with("CATEGORY", (mCategoryPaths.has(p.category) ? mCategoryPaths[p.category].asString() : mCategoryPaths["all"].asString()));
	}

	// add the search query string
	subs["QUERY"] = LLURI::escape(p.query);

	// add the permissions token that login.cgi gave us
	// We use "search_token", and fallback to "auth_token" if not present.
	LLSD search_token = LLLoginInstance::getInstance()->getResponse("search_token");
	if (search_token.asString().empty())
	{
		search_token = LLLoginInstance::getInstance()->getResponse("auth_token");
	}
	subs["AUTH_TOKEN"] = search_token.asString();

	// add the user's preferred maturity (can be changed via prefs)
	std::string maturity;

	// on Second Life grids, the maturity level is now a "&maturity" parameter that's not in the provided search URL
	if (LLGridManager::getInstance()->isInSecondLife())
	{
		if (gAgent.prefersAdult())
		{
			maturity = "gma";  // PG,Mature,Adult
		}
		else if (gAgent.prefersMature())
		{
			maturity = "gm";  // PG,Mature
		}
		else
		{
			maturity = "g";  // PG
		}

		// not used on the SL search anymore, so clear out the respective parameter
		subs["MATURITY"] = "";
	}
	// OpenSim probably still uses the old maturity variant, so keep the old behavior here
	else
	{
		if (gAgent.prefersAdult())
		{
			maturity = "42";  // PG,Mature,Adult
		}
		else if (gAgent.prefersMature())
		{
			maturity = "21";  // PG,Mature
		}
		else
		{
			maturity = "13";  // PG
		}
		subs["MATURITY"] = maturity;
	}

	// add the user's god status
	subs["GODLIKE"] = gAgent.isGodlike() ? "1" : "0";

	// Get the search URL and expand all of the substitutions
	// (also adds things like [LANGUAGE], [VERSION], [OS], etc.)
	std::string url;

	// add the maturity and category variables to the new Second Life search URL
	if (LLGridManager::getInstance()->isInSecondLife())
	{
		url = LFSimFeatureHandler::instance().searchURL() + "&maturity=" + maturity + "&" + mCategoryPaths[p.category].asString();
	}
	// for OpenSim, do the same as in earlier versions
	else
	{
		std::string debug_url = gSavedSettings.getString("SearchURLDebug");
		if (gSavedSettings.getBOOL("DebugSearch") && !debug_url.empty())
		{
			url = debug_url;
		}
		else
		{
			url = LFSimFeatureHandler::instance().searchURL();
		}
	}

	url = LLWeb::expandURLSubstitutions(url, subs);

	// Finally, load the URL in the webpanel
	mWebBrowser->navigateTo(url, HTTP_CONTENT_TEXT_HTML);
}

void FSPanelSearchWeb::focusDefaultElement()
{
	mWebBrowser->setFocus(TRUE);
}

void FSPanelSearchWeb::draw()
{
	if (mResetFocusOnLoad)
	{
		focusDefaultElement();
		mResetFocusOnLoad = false;
	}

	FSSearchPanelBase::draw();
}

////////////////////////////////////////
//           Local functions          //
////////////////////////////////////////

std::string filterShortWords(std::string query_string)
{
	if (query_string.length() < 1)
	{
		return "";
	}

	std::string final_query;
	bool filtered = false;
	boost::char_separator<char> sep(" ");
	boost::tokenizer<boost::char_separator<char> > tokens(query_string, sep);
	boost::tokenizer<boost::char_separator<char> >::iterator iter = tokens.begin();
	boost::tokenizer<boost::char_separator<char> >::iterator last = tokens.end();
	boost::tokenizer<boost::char_separator<char> >::iterator temp;
	for (; iter != last; ++iter)
	{
		if ((*iter).length() > MIN_SEARCH_STRING_SIZE)
		{
			final_query.append((*iter));
			temp = iter; ++temp;
			if (temp != last)
			{
				final_query.append(" ");
			}
		}
		else
		{
			filtered = true;
		}
	}

	if (filtered)
	{
		LLSD args = LLSD().with("FINALQUERY", final_query);
		LLNotificationsUtil::add("SeachFilteredOnShortWords", args);
	}
	
	return final_query;
}

void fillSearchComboBox(LLSearchComboBox* search_combo)
{
	if (search_combo == nullptr)
	{
		return;
	}

	LLSearchHistory::getInstance()->load();

	LLSearchHistory::search_history_list_t search_list =
	LLSearchHistory::getInstance()->getSearchHistoryList();
	LLSearchHistory::search_history_list_t::const_iterator it = search_list.begin();
	for ( ; search_list.end() != it; ++it)
	{
		LLSearchHistory::LLSearchHistoryItem item = *it;
		search_combo->add(item.search_query);
	}
}
