/**
 * @file fsfloatersearch.h
 * @brief Firestorm search definitions
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

#ifndef FS_FLOATERSEARCH_H
#define FS_FLOATERSEARCH_H

#include "llfloater.h"
#include "lliconctrl.h"
#include "lltexteditor.h"
#include "lltexturectrl.h"
#include "llremoteparcelrequest.h"
#include "llavatarpropertiesprocessor.h"
#include "llgroupmgr.h"
#include "llavatarnamecache.h"
#include "llmediactrl.h"
#include "llradiogroup.h"
#include "llsearchcombobox.h"
#include "llscrolllistctrl.h"
#include "lltabcontainer.h"

class FSSearchRemoteParcelInfoObserver;
class LLAvatarPropertiesObserver;
class LLGroupMgrObserver;
class LLSearchEditor;
class LLSearchComboBox;
class FSFloaterSearch;
class LLPanelProfile;
class FSScrollListCtrl;

struct SearchQuery : public LLInitParam::Block<SearchQuery>
{
	Optional<std::string> category;
	Optional<std::string> query;

	SearchQuery();
};

///////////////////////////////
//       Search Panels       //
///////////////////////////////

class FSSearchPanelBase : public LLPanel
{
public:
	FSSearchPanelBase() : LLPanel() { }
	virtual ~FSSearchPanelBase() { }
	virtual void focusDefaultElement() { }
};

class FSPanelSearchPeople : public FSSearchPanelBase
{
	LOG_CLASS(FSFloaterSearch);
public:
	FSPanelSearchPeople();
	static void processSearchReply(LLMessageSystem* msg, void**);

	/*virtual*/ void focusDefaultElement();

protected:
	const S32& getNumResultsReturned() const { return mNumResultsReturned; };
	const S32& getNumResultsReceived() const { return mResultsReceived; };

private:
	/*virtual*/ BOOL postBuild();
	virtual ~FSPanelSearchPeople();
	
	void onBtnFind();
	void onSelectItem();
	void onBtnNext();
	void onBtnBack();

	void find();
	void resetSearch();
	S32  showNextButton(S32);

	const LLUUID& getQueryID() const { return mQueryID; }

	void onAvatarNameCallback(const LLUUID& id, const LLAvatarName& av_name);

	typedef boost::signals2::connection avatar_name_callback_connection_t;
	avatar_name_callback_connection_t mAvatarNameCallbackConnection;

	S32			mNumResultsReturned;
	S32			mStartSearch;
	S32			mResultsReceived;
	LLSD		mResultsContent;
	LLUUID		mQueryID;

	FSFloaterSearch*		mParent;
	LLSearchComboBox*		mSearchComboBox;
	FSScrollListCtrl*		mSearchResults;
};

class FSPanelSearchGroups : public FSSearchPanelBase
{
	LOG_CLASS(FSFloaterSearch);
public:
	FSPanelSearchGroups();
	static void processSearchReply(LLMessageSystem* msg, void**);

	/*virtual*/ void focusDefaultElement();

private:
	/*virtual*/ BOOL postBuild();
	virtual ~FSPanelSearchGroups();

	void onBtnFind();
	void onSelectItem();
	void onBtnNext();
	void onBtnBack();

	void find();
	void resetSearch();
	S32  showNextButton(S32);

	const LLUUID& getQueryID() const { return mQueryID; }

	S32			mNumResultsReturned;
	S32			mStartSearch;
	S32			mResultsReceived;
	LLSD		mResultsContent;
	LLUUID		mQueryID;

	FSFloaterSearch*	mParent;
	LLSearchComboBox*	mSearchComboBox;
	LLScrollListCtrl*	mSearchResults;
};

class FSPanelSearchPlaces : public FSSearchPanelBase
{
	LOG_CLASS(FSFloaterSearch);
public:
	FSPanelSearchPlaces();
	static void processSearchReply(LLMessageSystem* msg, void**);

	/*virtual*/ void focusDefaultElement();

private:
	/*virtual*/ BOOL postBuild();
	virtual ~FSPanelSearchPlaces();

	void onBtnFind();
	void onSelectItem();
	void onBtnNext();
	void onBtnBack();

	void find();
	void resetSearch();
	S32  showNextButton(S32);

	const LLUUID& getQueryID() const { return mQueryID; }
	
	S32			mNumResultsReturned;
	S32			mStartSearch;
	S32			mResultsReceived;
	LLSD		mResultsContent;
	LLUUID		mQueryID;

	FSFloaterSearch*	mParent;
	LLSearchComboBox*	mSearchComboBox;
	LLScrollListCtrl*	mSearchResults;
	LLComboBox*			mPlacesCategory;
};

class FSPanelSearchLand : public FSSearchPanelBase
{
	LOG_CLASS(FSFloaterSearch);
public:
	FSPanelSearchLand();
	static void processSearchReply(LLMessageSystem* msg, void**);

private:
	/*virtual*/ BOOL postBuild();
	virtual ~FSPanelSearchLand();

	void onBtnFind();
	void onSelectItem();
	void onBtnNext();
	void onBtnBack();

	void find();
	void resetSearch();
	S32  showNextButton(S32);

	const LLUUID& getQueryID() const { return mQueryID; }

	S32			mNumResultsReturned;
	S32			mStartSearch;
	S32			mResultsReceived;
	LLSD		mResultsContent;
	LLUUID		mQueryID;

	FSFloaterSearch*	mParent;
	LLLineEditor*		mPriceEditor;
	LLLineEditor*		mAreaEditor;
	LLScrollListCtrl*	mSearchResults;
};

class FSPanelSearchClassifieds : public FSSearchPanelBase
{
	LOG_CLASS(FSFloaterSearch);
public:
	FSPanelSearchClassifieds();
	static void processSearchReply(LLMessageSystem* msg, void**);
	
	/*virtual*/ void focusDefaultElement();

private:
	/*virtual*/ BOOL postBuild();
	virtual ~FSPanelSearchClassifieds();

	void onBtnFind();
	void onSelectItem();
	void onBtnNext();
	void onBtnBack();

	void find();
	void resetSearch();
	S32  showNextButton(S32);

	const LLUUID& getQueryID() const { return mQueryID; }
	
	S32			mNumResultsReturned;
	S32			mStartSearch;
	S32			mResultsReceived;
	LLSD		mResultsContent;
	LLUUID		mQueryID;

	FSFloaterSearch*	mParent;
	LLSearchComboBox*	mSearchComboBox;
	LLScrollListCtrl*	mSearchResults;
	LLComboBox*			mClassifiedsCategory;
};

class FSPanelSearchEvents : public FSSearchPanelBase
{
	LOG_CLASS(FSFloaterSearch);
public:
	FSPanelSearchEvents();
	static void processSearchReply(LLMessageSystem* msg, void**);

	/*virtual*/ void focusDefaultElement();

private:
	/*virtual*/ BOOL postBuild();
	virtual ~FSPanelSearchEvents();

	void onBtnFind();
	void onSelectItem();
	void onBtnNext();
	void onBtnBack();
	void onBtnTomorrow();
	void onBtnYesterday();
	void onBtnToday();
	
	void find();
	void setDay(S32 day);
	void onSearchModeChanged();
	void resetSearch();
	S32  showNextButton(S32);

	const LLUUID& getQueryID() const { return mQueryID; }

	S32			mNumResultsReturned;
	S32			mResultsReceived;
	S32			mStartSearch;
	S32			mDay;
	LLSD		mResultsContent;
	LLUUID		mQueryID;

	FSFloaterSearch*	mParent;
	LLSearchComboBox*	mSearchComboBox;
	LLScrollListCtrl*	mSearchResults;
	LLRadioGroup*		mEventsMode;
};

class FSPanelSearchWeb : public FSSearchPanelBase
{
	LOG_CLASS(FSFloaterSearch);
public:
	FSPanelSearchWeb();
	/*virtual*/ BOOL postBuild();
	void loadURL(const SearchQuery &query);
	/*virtual*/ void focusDefaultElement();
	/*virtual*/ void draw();
	void resetFocusOnLoad() { mResetFocusOnLoad = true; }

private:
	virtual ~FSPanelSearchWeb() {};

	LLMediaCtrl*	mWebBrowser;
	LLSD			mCategoryPaths;

	bool			mResetFocusOnLoad;
};

class FSFloaterSearch : public LLFloater
{
	LOG_CLASS(FSFloaterSearch);
public:
	typedef enum e_search_category
	{
		SC_AVATAR,
		SC_GROUP,
		SC_PLACE,
		SC_CLASSIFIED
	}	ESearchCategory;

	struct _Params : public LLInitParam::Block<_Params, LLFloater::Params>
	{
		Optional<SearchQuery> search;
	};

	typedef LLSDParamAdapter<_Params> Params;
	
	FSFloaterSearch(const Params& key);
	~FSFloaterSearch();
	void onOpen(const LLSD& key);
	BOOL postBuild();

	void avatarNameUpdatedCallback(const LLUUID& id, const LLAvatarName& av_name);
	void groupNameUpdatedCallback(const LLUUID& id, const std::string& name, bool is_group);
	void onSelectedItem(const LLUUID& selected_item, ESearchCategory type);
	void onSelectedEvent(const S32 selected_event);
	void displayParcelDetails(const LLParcelData& parcel_data);
	void displayClassifiedDetails(LLAvatarClassifiedInfo*& c_info);
	void displayAvatarDetails(LLAvatarData*& avatar_data);
	void displayGroupDetails(LLGroupMgrGroupData*& group_data);
	void displayEventDetails(U32 eventId,
							 F64 eventEpoch,
							 const std::string& eventDateStr,
							 const std::string &eventName,
							 const std::string &eventDesc,
							 const std::string &simName,
							 U32 eventDuration,
							 U32 eventFlags,
							 U32 eventCover,
							 LLVector3d eventGlobalPos);
	void displayEventParcelImage(const LLParcelData& parcel_data);
	void setLoadingProgress(bool started);

	template <class T>
	static T* getSearchPanel(const std::string& panel_name);

private:
	virtual void onClose(bool app_quitting);
	const LLUUID& getSelectedID() { return mSelectedID; }
	LLVector3d	mParcelGlobal;
	LLUUID		mSelectedID;
	U32			mEventID;
	bool		mHasSelection;

	void resetVerbs();
	void flushDetails();
	void onTabChange();
	void onBtnPeopleProfile();
	void onBtnPeopleIM();
	void onBtnPeopleFriend();
	void onBtnGroupProfile();
	void onBtnGroupChat();
	void onBtnGroupJoin();
	void onBtnEventReminder();
	void onBtnTeleport();
	void onBtnMap();

	void regionHandleCallback(U64 region_handle, LLVector3d pos_global);

	FSSearchRemoteParcelInfoObserver* mRemoteParcelObserver;
	FSSearchRemoteParcelInfoObserver* mRemoteParcelEventLocationObserver;
	LLAvatarPropertiesObserver* mAvatarPropertiesObserver;
	LLGroupMgrObserver* mGroupPropertiesRequest;

	FSPanelSearchPeople*	mPanelPeople;
	FSPanelSearchGroups*	mPanelGroups;
	FSPanelSearchPlaces*	mPanelPlaces;
	FSPanelSearchEvents*	mPanelEvents;
	FSPanelSearchLand*		mPanelLand;
	FSPanelSearchClassifieds* mPanelClassifieds;
	FSPanelSearchWeb*		mPanelWeb;

	LLPanel*		mDetailsPanel;
	LLTextEditor*	mDetailTitle;
	LLTextEditor*	mDetailDesc;
	LLTextEditor*	mDetailAux1;
	LLTextEditor*	mDetailAux2;
	LLTextEditor*	mDetailLocation;
	LLTextureCtrl*	mDetailSnapshot;
	LLTextureCtrl*	mDetailSnapshotParcel;
	LLIconCtrl*		mDetailMaturity;
	LLTabContainer*	mTabContainer;
};

#endif // FS_FLOATERSEARCH_H
