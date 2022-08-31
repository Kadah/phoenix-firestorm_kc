/**
 * @file fsfloaterplacedetails.cpp
 * @brief Class for the place details floater in Firestorm
 *
 * $LicenseInfo:firstyear=2012&license=viewerlgpl$
 * Phoenix Firestorm Viewer Source Code
 * Copyright (c) 2012 Ansariel Hiller @ Second Life
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

#include "fsfloaterplacedetails.h"

#include "llbutton.h"
#include "llcombobox.h"
#include "lllineeditor.h"
#include "llmenubutton.h"
#include "lltexteditor.h"
#include "lltoggleablemenu.h"
#include "llwindow.h"

#include "llagent.h"
#include "llagentpicksinfo.h"
#include "llfloaterreg.h"
#include "llfloatersidepanelcontainer.h"
#include "llfloaterworldmap.h"
#include "llinventoryobserver.h"
#include "lllandmarkactions.h"
#include "lllandmarklist.h"
#include "llnotificationsutil.h"
#include "llpanelplaceprofile.h"
#include "llpanellandmarkinfo.h"
#include "llparcel.h"
#include "llregionhandle.h" // <FS:Beq/> Var region support
#include "llteleporthistorystorage.h"
#include "llviewercontrol.h"
#include "llviewermessage.h"
#include "llviewermenu.h"
#include "llviewerparcelmgr.h"
#include "llviewerwindow.h"
#include "llviewerinventory.h"
#include "rlvactions.h"

static const F32 FS_PLACE_INFO_UPDATE_INTERVAL = 3.0f;

static bool fs_is_agent_in_selected_parcel(LLParcel* parcel)
{
	LLViewerParcelMgr* parcel_mgr = LLViewerParcelMgr::getInstance();

	LLViewerRegion* region = parcel_mgr->getSelectionRegion();
	if (!region || !parcel)
		return false;

	return	region == gAgent.getRegion() &&
			parcel->getLocalID() == parcel_mgr->getAgentParcel()->getLocalID();
}


/////////////////////////////////
// Inventory added observer
/////////////////////////////////

class FSPlaceDetailsInventoryObserver : public LLInventoryAddedObserver
{
public:
	FSPlaceDetailsInventoryObserver(FSFloaterPlaceDetails* place_details_floater) :
		mPlaceDetails(place_details_floater)
	{}

protected:
	/*virtual*/ void done()
	{
		mPlaceDetails->showAddedLandmarkInfo(gInventory.getAddedIDs());
	}

private:
	FSFloaterPlaceDetails*	mPlaceDetails;
};


/////////////////////////////////
// Remote Parcel info observer
/////////////////////////////////

class FSPlaceDetailsRemoteParcelInfoObserver : public LLRemoteParcelInfoObserver
{
public:
	FSPlaceDetailsRemoteParcelInfoObserver(FSFloaterPlaceDetails* place_details_floater) :
		LLRemoteParcelInfoObserver(),
		mPlaceDetails(place_details_floater)
	{}

	~FSPlaceDetailsRemoteParcelInfoObserver()
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
		if (mPlaceDetails)
		{
			mPlaceDetails->changedGlobalPos(LLVector3d(parcel_data.global_x,
												       parcel_data.global_y,
												       parcel_data.global_z));

			mPlaceDetails->processParcelDetails(parcel_data);
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
		LL_WARNS() << "Can't complete remote parcel request. Http Status: "
			    << status << ". Reason : " << reason << LL_ENDL;
	}

private:
	std::set<LLUUID>		mParcelIDs;
	FSFloaterPlaceDetails*	mPlaceDetails;
};


/////////////////////////////////
// Parcel info observer
/////////////////////////////////

class FSPlaceDetailsPlacesParcelObserver : public LLParcelObserver
{
public:
	FSPlaceDetailsPlacesParcelObserver(FSFloaterPlaceDetails* place_details_floater) :
		LLParcelObserver(),
		mPlaceDetails(place_details_floater)
	{}

	/*virtual*/ void changed()
	{
		if (mPlaceDetails)
		{
			mPlaceDetails->changedParcelSelection();
		}
	}

private:
	FSFloaterPlaceDetails*		mPlaceDetails;
};

///////////////////////////////////////
// FSFloaterPlaceDetails implementation
///////////////////////////////////////

FSFloaterPlaceDetails::FSFloaterPlaceDetails(const LLSD& seed)
	: LLFloater(seed),
	mPanelLandmarkInfo(NULL),
	mPanelPlaceInfo(NULL),
	mIsInEditMode(false),
	mIsInCreateMode(false),
	mGlobalPos(),
	mDisplayInfo(NONE)
{
	mParcelObserver = new FSPlaceDetailsPlacesParcelObserver(this);
	mRemoteParcelObserver = new FSPlaceDetailsRemoteParcelInfoObserver(this);
	mInventoryObserver = new FSPlaceDetailsInventoryObserver(this);
	gInventory.addObserver(mInventoryObserver);

	mAgentParcelChangedConnection = gAgent.addParcelChangedCallback(
			boost::bind(&FSFloaterPlaceDetails::updateVerbs, this));
}

FSFloaterPlaceDetails::~FSFloaterPlaceDetails()
{
	if (gInventory.containsObserver(mInventoryObserver))
	{
		gInventory.removeObserver(mInventoryObserver);
	}
	LLViewerParcelMgr::getInstance()->removeObserver(mParcelObserver);
	mParcel.clear();

	delete mInventoryObserver;
	delete mRemoteParcelObserver;
	delete mParcelObserver;

	if (mAgentParcelChangedConnection.connected())
	{
		mAgentParcelChangedConnection.disconnect();
	}
}

BOOL FSFloaterPlaceDetails::postBuild()
{
	mPanelLandmarkInfo = findChild<LLPanelLandmarkInfo>("panel_landmark_info");
	mPanelPlaceInfo = findChild<LLPanelPlaceProfile>("panel_place_profile");

	if (!mPanelLandmarkInfo || !mPanelPlaceInfo)
	{
		return FALSE;
	}

	getChild<LLButton>("teleport_btn")->setClickedCallback(boost::bind(&FSFloaterPlaceDetails::onTeleportButtonClicked, this));
	getChild<LLButton>("map_btn")->setClickedCallback(boost::bind(&FSFloaterPlaceDetails::onShowOnMapButtonClicked, this));
	getChild<LLButton>("edit_btn")->setClickedCallback(boost::bind(&FSFloaterPlaceDetails::onEditButtonClicked, this));
	getChild<LLButton>("save_btn")->setClickedCallback(boost::bind(&FSFloaterPlaceDetails::onSaveButtonClicked, this));
	getChild<LLButton>("cancel_btn")->setClickedCallback(boost::bind(&FSFloaterPlaceDetails::onCancelButtonClicked, this));
	getChild<LLButton>("close_btn")->setClickedCallback(boost::bind(&FSFloaterPlaceDetails::onCloseButtonClicked, this));

	LLLineEditor* title_editor = mPanelLandmarkInfo->getChild<LLLineEditor>("title_editor");
	title_editor->setKeystrokeCallback(boost::bind(&FSFloaterPlaceDetails::onEditButtonClicked, this), NULL);

	LLTextEditor* notes_editor = mPanelLandmarkInfo->getChild<LLTextEditor>("notes_editor");
	notes_editor->setKeystrokeCallback(boost::bind(&FSFloaterPlaceDetails::onEditButtonClicked, this));

	LLComboBox* folder_combo = mPanelLandmarkInfo->getChild<LLComboBox>("folder_combo");
	folder_combo->setCommitCallback(boost::bind(&FSFloaterPlaceDetails::onEditButtonClicked, this));

	mOverflowBtn = getChild<LLMenuButton>("overflow_btn");
	mOverflowBtn->setMouseDownCallback(boost::bind(&FSFloaterPlaceDetails::onOverflowButtonClicked, this));

	LLUICtrl::CommitCallbackRegistry::ScopedRegistrar registrar;
	registrar.add("Places.OverflowMenu.Action",  boost::bind(&FSFloaterPlaceDetails::onOverflowMenuItemClicked, this, _2));
	LLUICtrl::EnableCallbackRegistry::ScopedRegistrar enable_registrar;
	enable_registrar.add("Places.OverflowMenu.Enable",  boost::bind(&FSFloaterPlaceDetails::onOverflowMenuItemEnable, this, _2));

	mPlaceMenu = LLUICtrlFactory::getInstance()->createFromFile<LLToggleableMenu>("menu_place.xml", gMenuHolder, LLViewerMenuHolderGL::child_registry_t::instance());
	if (!mPlaceMenu)
	{
		LL_WARNS() << "Error loading Place menu" << LL_ENDL;
	}

	mLandmarkMenu = LLUICtrlFactory::getInstance()->createFromFile<LLToggleableMenu>("menu_landmark.xml", gMenuHolder, LLViewerMenuHolderGL::child_registry_t::instance());
	if (!mLandmarkMenu)
	{
		LL_WARNS() << "Error loading Landmark menu" << LL_ENDL;
	}

	updateVerbs();

	return TRUE;
}

void FSFloaterPlaceDetails::onOpen(const LLSD& key)
{
	mIsInCreateMode = false;
	mIsInEditMode = false;

	if (key.size() != 0)
	{
		std::string key_type = key["type"].asString();

		if (key_type == "landmark")
		{
			mDisplayInfo = LANDMARK;
			setTitle(getString("title_landmark"));

			LLInventoryItem* item = gInventory.getItem(key["id"].asUUID());
			if (!item)
			{
				return;
			}

			mPanelLandmarkInfo->resetLocation();
			mPanelLandmarkInfo->setInfoType(LLPanelPlaceInfo::LANDMARK);
			mPanelLandmarkInfo->setHeaderVisible(FALSE);
			
			mPanelPlaceInfo->setVisible(FALSE);
			mPanelLandmarkInfo->setVisible(TRUE);

			setItem(item);
		}
		else if (key_type == "create_landmark")
		{
			mDisplayInfo = CREATE_LANDMARK;
			setTitle(getString("title_create_landmark"));

			if (key.has("x") && key.has("y") && key.has("z"))
			{
				mGlobalPos = LLVector3d(key["x"].asReal(),
										key["y"].asReal(),
										key["z"].asReal());
			}
			else
			{
				mGlobalPos = gAgent.getPositionGlobal();
			}

			LLUUID dest_folder = key["dest_folder"];
			mPanelLandmarkInfo->resetLocation();
			mPanelLandmarkInfo->setInfoAndCreateLandmark(dest_folder);
			mPanelLandmarkInfo->setHeaderVisible(FALSE);
			mPanelLandmarkInfo->displayParcelInfo(LLUUID(), mGlobalPos);
			
			mPanelPlaceInfo->setVisible(FALSE);
			mPanelLandmarkInfo->setVisible(TRUE);

			mIsInCreateMode = true;
			updateVerbs();
		}
		else if (key_type == "remote_place")
		{
			mDisplayInfo = REMOTE_PLACE;
			setTitle(getString("title_remote_place"));

			mPanelPlaceInfo->resetLocation();
			mPanelPlaceInfo->setInfoType(LLPanelPlaceInfo::PLACE);
			mPanelPlaceInfo->setHeaderVisible(FALSE);
			mPanelPlaceInfo->setVisible(TRUE);
			mPanelLandmarkInfo->setVisible(FALSE);

			if (key.has("id"))
			{
				LLUUID parcel_id = key["id"].asUUID();
				mPanelPlaceInfo->setParcelID(parcel_id);

				// query the server to get the global 3D position of this
				// parcel - we need this for teleport/mapping functions.
				mRemoteParcelObserver->setParcelID(parcel_id);
			}
			else
			{
				mGlobalPos = LLVector3d(key["x"].asReal(),
										key["y"].asReal(),
										key["z"].asReal());

				mPanelPlaceInfo->setParcelDetailLoadedCallback(boost::bind(&FSFloaterPlaceDetails::processParcelDetails, this, _1));
				if(key.has("ox"))
				{
					auto region_handle = to_region_handle(key["ox"].asInteger(), key["oy"].asInteger());
					mPanelPlaceInfo->displayParcelInfo(LLUUID(), region_handle, mGlobalPos);
				}
				else
				{
					mPanelPlaceInfo->displayParcelInfo(LLUUID(), mGlobalPos);
				}
			}

			updateVerbs();
		}
		else if (key_type == "teleport_history")
		{
			mDisplayInfo = TELEPORT_HISTORY_ITEM;

			S32 index = key["id"].asInteger();

			const LLTeleportHistoryStorage::slurl_list_t& hist_items =
						LLTeleportHistoryStorage::getInstance()->getItems();

			mGlobalPos = hist_items[index].mGlobalPos;

			LLStringUtil::format_map_t args;
			args["[NAME]"] = hist_items[index].mTitle.c_str();
			setTitle(getString("title_teleport_history_item", args));

			mPanelPlaceInfo->resetLocation();
			mPanelPlaceInfo->setInfoType(LLPanelPlaceInfo::TELEPORT_HISTORY);
			mPanelPlaceInfo->displayParcelInfo(LLUUID(), mGlobalPos);

			mPanelPlaceInfo->setHeaderVisible(FALSE);
			mPanelPlaceInfo->setVisible(TRUE);
			mPanelLandmarkInfo->setVisible(FALSE);

			updateVerbs();
		}
		else if (key_type == "agent")
		{
			mDisplayInfo = AGENT;

			setTitle(getString("title_place"));

			mPanelPlaceInfo->resetLocation();
			mPanelPlaceInfo->setInfoType(LLPanelPlaceInfo::AGENT);
			mPanelPlaceInfo->setHeaderVisible(FALSE);
			mPanelPlaceInfo->setVisible(TRUE);
			mPanelLandmarkInfo->setVisible(FALSE);

			LLViewerParcelMgr* parcel_mgr = LLViewerParcelMgr::getInstance();
			if (!parcel_mgr)
			{
				return;
			}
			// We don't know if we are already added to LLViewerParcelMgr observers list
			// so try to remove observer not to add an extra one.
			parcel_mgr->removeObserver(mParcelObserver);
			parcel_mgr->addObserver(mParcelObserver);
			parcel_mgr->selectParcelAt(gAgent.getPositionGlobal());
		}
	}
}


void FSFloaterPlaceDetails::updateVerbs()
{
	if (mDisplayInfo == NONE)
	{
		getChildView("teleport_btn")->setVisible(!(mIsInEditMode || mIsInCreateMode));
		getChildView("map_btn")->setVisible(!(mIsInEditMode || mIsInCreateMode));
		getChildView("edit_btn")->setVisible(!(mIsInEditMode || mIsInCreateMode));
		getChildView("save_btn")->setVisible(mIsInEditMode);
		getChildView("cancel_btn")->setVisible(mIsInEditMode);
		getChildView("close_btn")->setVisible(mIsInCreateMode);

		return;
	}

	bool have_position = !mGlobalPos.isExactlyZero();
	getChildView("teleport_btn")->setEnabled(have_position);
	getChildView("map_btn")->setEnabled(have_position);

	if (mDisplayInfo == CREATE_LANDMARK || mDisplayInfo == LANDMARK)
	{
		getChildView("teleport_btn")->setVisible(!(mIsInEditMode || mIsInCreateMode));
		getChildView("map_btn")->setVisible(!(mIsInEditMode || mIsInCreateMode));
		getChildView("edit_btn")->setVisible(!(mIsInEditMode || mIsInCreateMode));
		getChildView("save_btn")->setVisible(mIsInEditMode);
		getChildView("cancel_btn")->setVisible(mIsInEditMode);
		getChildView("close_btn")->setVisible(mIsInCreateMode);
		mOverflowBtn->setVisible(!(mIsInEditMode || mIsInCreateMode));
	}
	else if (mDisplayInfo == REMOTE_PLACE || mDisplayInfo == TELEPORT_HISTORY_ITEM)
	{
		getChildView("teleport_btn")->setVisible(TRUE);
		getChildView("teleport_btn")->setEnabled(RlvActions::canTeleportToLocation());
		getChildView("map_btn")->setVisible(TRUE);
		getChildView("edit_btn")->setVisible(FALSE);
		getChildView("save_btn")->setVisible(FALSE);
		getChildView("cancel_btn")->setVisible(FALSE);
		getChildView("close_btn")->setVisible(FALSE);
	}
	else if (mDisplayInfo == AGENT)
	{
		getChildView("teleport_btn")->setVisible(TRUE);
		getChildView("teleport_btn")->setEnabled(have_position && !LLViewerParcelMgr::getInstance()->inAgentParcel(mGlobalPos));
		getChildView("map_btn")->setVisible(TRUE);
		getChildView("edit_btn")->setVisible(FALSE);
		getChildView("save_btn")->setVisible(FALSE);
		getChildView("cancel_btn")->setVisible(FALSE);
		getChildView("close_btn")->setVisible(FALSE);
	}
}



void FSFloaterPlaceDetails::showAddedLandmarkInfo(const uuid_set_t& items)
{
	for (uuid_set_t::const_iterator item_iter = items.begin();
		 item_iter != items.end();
		 ++item_iter)
	{
		const LLUUID& item_id = (*item_iter);
		if(!highlight_offered_object(item_id))
		{
			continue;
		}

		LLInventoryItem* item = gInventory.getItem(item_id);

		llassert(item);
		if (item && (LLAssetType::AT_LANDMARK == item->getType()) )
		{
			// Created landmark is passed to Places panel to allow its editing.
			// If the panel is closed we don't reopen it until created landmark is loaded.
			//if("create_landmark" == getPlaceInfoType() && !getItem())
			//{
				setItem(item);
			//}
		}
	}
}

void FSFloaterPlaceDetails::setItem(LLInventoryItem* item)
{
	if (!item)
	{
		return;
	}

	if (mDisplayInfo == LANDMARK)
	{
		LLStringUtil::format_map_t args;
		args["[NAME]"] = item->getName().c_str();
		setTitle(getString("title_landmark_detail", args));
	}

	mItem = item;

	LLAssetType::EType item_type = mItem->getActualType();
	if (item_type == LLAssetType::AT_LANDMARK || item_type == LLAssetType::AT_LINK)
	{
		// If the item is a link get a linked item
		if (item_type == LLAssetType::AT_LINK)
		{
			mItem = gInventory.getItem(mItem->getLinkedUUID());
			if (mItem.isNull())
			{
				return;
			}
		}
	}
	else
	{
		return;
	}

	// Check if item is in agent's inventory and he has the permission to modify it.
	BOOL is_landmark_editable = gInventory.isObjectDescendentOf(mItem->getUUID(), gInventory.getRootFolderID()) &&
								mItem->getPermissions().allowModifyBy(gAgent.getID());

	getChildView("edit_btn")->setEnabled(is_landmark_editable);
	getChildView("save_btn")->setEnabled(is_landmark_editable);

	if (is_landmark_editable)
	{
		if(!mPanelLandmarkInfo->setLandmarkFolder(mItem->getParentUUID()) && !mItem->getParentUUID().isNull())
		{
			const LLViewerInventoryCategory* cat = gInventory.getCategory(mItem->getParentUUID());
			if (cat)
			{
				std::string cat_fullname = LLPanelLandmarkInfo::getFullFolderName(cat);
				LLComboBox* folderList = mPanelLandmarkInfo->getChild<LLComboBox>("folder_combo");
				folderList->add(cat_fullname, cat->getUUID(), ADD_TOP);
			}
		}
	}

	mPanelLandmarkInfo->displayItemInfo(mItem);

	LLLandmark* lm = gLandmarkList.getAsset(mItem->getAssetUUID(),
											boost::bind(&FSFloaterPlaceDetails::onLandmarkLoaded, this, _1));
	if (lm)
	{
		onLandmarkLoaded(lm);
	}
}

// static
void FSFloaterPlaceDetails::showPlaceDetails(const LLSD& key)
{
	if (gSavedSettings.getBOOL("FSUseStandalonePlaceDetailsFloater"))
	{
		LLFloaterReg::showInstance("fs_placedetails", key);
	}
	else
	{
		LLFloaterSidePanelContainer::showPanel("places", key);
	}
}

/////////////////////////////////
// Button event handlers
/////////////////////////////////

void FSFloaterPlaceDetails::onTeleportButtonClicked()
{
	if (mDisplayInfo == LANDMARK || mDisplayInfo == CREATE_LANDMARK)
	{
		if (mItem.isNull())
		{
			return;
		}

		LLSD payload;
		payload["asset_id"] = mItem->getAssetUUID();
		LLSD args; 
		args["LOCATION"] = mItem->getName(); 
		LLNotificationsUtil::add("TeleportFromLandmark", args, payload);
	}
	else if (mDisplayInfo == REMOTE_PLACE || mDisplayInfo == TELEPORT_HISTORY_ITEM || mDisplayInfo == AGENT)
	{
		LLFloaterWorldMap* worldmap_instance = LLFloaterWorldMap::getInstance();
		if (!mGlobalPos.isExactlyZero() && worldmap_instance)
		{
			gAgent.teleportViaLocation(mGlobalPos);
			worldmap_instance->trackLocation(mGlobalPos);
		}
	}
}

void FSFloaterPlaceDetails::onShowOnMapButtonClicked()
{
	LLFloaterWorldMap* worldmap_instance = LLFloaterWorldMap::getInstance();
	if(!worldmap_instance)
		return;

	if (mDisplayInfo == LANDMARK || mDisplayInfo == CREATE_LANDMARK)
	{
		LLLandmark* landmark = gLandmarkList.getAsset(mItem->getAssetUUID());
		if (!landmark)
			return;

		LLVector3d landmark_global_pos;
		if (!landmark->getGlobalPos(landmark_global_pos))
			return;
			
		if (!landmark_global_pos.isExactlyZero())
		{
			worldmap_instance->trackLocation(landmark_global_pos);
			LLFloaterReg::showInstance("world_map", "center");
		}
	}
	else if (mDisplayInfo == REMOTE_PLACE || mDisplayInfo == TELEPORT_HISTORY_ITEM || mDisplayInfo == AGENT)
	{
		if (!mGlobalPos.isExactlyZero())
		{
			worldmap_instance->trackLocation(mGlobalPos);
			LLFloaterReg::showInstance("world_map", "center");
		}
	}
}

void FSFloaterPlaceDetails::onEditButtonClicked()
{
	mPanelLandmarkInfo->toggleLandmarkEditMode(TRUE);
	mIsInEditMode = true;
	mIsInCreateMode = false;
	updateVerbs();
}

void FSFloaterPlaceDetails::onCancelButtonClicked()
{
	mPanelLandmarkInfo->toggleLandmarkEditMode(FALSE);
	mIsInEditMode = false;

	updateVerbs();

	// Reload the landmark properties.
	mPanelLandmarkInfo->displayItemInfo(mItem);
}

void FSFloaterPlaceDetails::onSaveButtonClicked()
{
	if (mItem.isNull())
	{
		return;
	}

	std::string current_title_value = mPanelLandmarkInfo->getLandmarkTitle();
	std::string item_title_value = mItem->getName();
	std::string current_notes_value = mPanelLandmarkInfo->getLandmarkNotes();
	std::string item_notes_value = mItem->getDescription();

	LLStringUtil::trim(current_title_value);
	LLStringUtil::trim(current_notes_value);

	LLUUID item_id = mItem->getUUID();
	LLUUID folder_id = mPanelLandmarkInfo->getLandmarkFolder();

	LLPointer<LLViewerInventoryItem> new_item = new LLViewerInventoryItem(mItem);

	if (!current_title_value.empty() &&
		(item_title_value != current_title_value || item_notes_value != current_notes_value))
	{
		new_item->rename(current_title_value);
		new_item->setDescription(current_notes_value);
		new_item->updateServer(FALSE);
	}

	if(folder_id != mItem->getParentUUID())
	{
		LLInventoryModel::update_list_t update;
		LLInventoryModel::LLCategoryUpdate old_folder(mItem->getParentUUID(),-1);
		update.push_back(old_folder);
		LLInventoryModel::LLCategoryUpdate new_folder(folder_id, 1);
		update.push_back(new_folder);
		gInventory.accountForUpdate(update);

		new_item->setParent(folder_id);
		new_item->updateParentOnServer(FALSE);
	}

	gInventory.updateItem(new_item);
	gInventory.notifyObservers();

	onCancelButtonClicked();
}

void FSFloaterPlaceDetails::onCloseButtonClicked()
{
	onSaveButtonClicked();
	closeFloater();
}

void FSFloaterPlaceDetails::onOverflowButtonClicked()
{
	LLToggleableMenu* menu;

	if ((mDisplayInfo == TELEPORT_HISTORY_ITEM || mDisplayInfo == REMOTE_PLACE || mDisplayInfo == AGENT) && mPlaceMenu)
	{
		menu = mPlaceMenu;

		// STORM-411
		// Creating landmarks for remote locations is impossible.
		// So hide menu item "Make a Landmark" in "Teleport History Profile" panel.
		menu->setItemVisible("landmark", mDisplayInfo == AGENT);
		menu->arrangeAndClear();
	}
	else if ((mDisplayInfo == LANDMARK || mDisplayInfo == CREATE_LANDMARK) && mLandmarkMenu)
	{
		menu = mLandmarkMenu;

		BOOL is_landmark_removable = FALSE;
		if (mItem.notNull())
		{
			const LLUUID& item_id = mItem->getUUID();
			const LLUUID trash_id = gInventory.findCategoryUUIDForType(LLFolderType::FT_TRASH);
			is_landmark_removable = gInventory.isObjectDescendentOf(item_id, gInventory.getRootFolderID()) &&
									!gInventory.isObjectDescendentOf(item_id, trash_id);
		}

		menu->getChild<LLMenuItemCallGL>("delete")->setEnabled(is_landmark_removable);
	}
	else
	{
		return;
	}

	// TODO: What to do with the create pick stuff? Disabled for now...
	menu->getChild<LLMenuItemCallGL>("pick")->setVisible(FALSE);

	mOverflowBtn->setMenu(menu, LLMenuButton::MP_TOP_RIGHT);
}

void FSFloaterPlaceDetails::onOverflowMenuItemClicked(const LLSD& param)
{
	std::string item = param.asString();
	if (item == "landmark")
	{
		LLSD key;
		key["type"] = "create_landmark";
		key["x"] = mGlobalPos.mdV[VX];
		key["y"] = mGlobalPos.mdV[VY];
		key["z"] = mGlobalPos.mdV[VZ];

		LLFloaterReg::showInstance("fs_placedetails", key);
	}
	else if (item == "copy")
	{
		LLLandmarkActions::getSLURLfromPosGlobal(mGlobalPos, boost::bind(&FSFloaterPlaceDetails::onSLURLBuilt, this, _1));
	}
	else if (item == "delete")
	{
		gInventory.removeItem(mItem->getUUID());
		closeFloater();
	}
	else if (item == "pick")
	{
		LLPanelPlaceInfo* panel = mPanelPlaceInfo;
		if (mDisplayInfo == LANDMARK)
		{
			panel = mPanelLandmarkInfo;
		}

		if (panel)
		{
			panel->createPick(mGlobalPos);
		}
	}
	else if (item == "add_to_favbar")
	{
		if ( mItem.notNull() )
		{
			const LLUUID& favorites_id = gInventory.findCategoryUUIDForType(LLFolderType::FT_FAVORITE);
			if ( favorites_id.notNull() )
			{
				copy_inventory_item(gAgent.getID(),
									mItem->getPermissions().getOwner(),
									mItem->getUUID(),
									favorites_id,
									std::string(),
									LLPointer<LLInventoryCallback>(NULL));
				LL_INFOS() << "Copied inventory item #" << mItem->getUUID() << " to favorites." << LL_ENDL;
			}
		}
	}
}

bool FSFloaterPlaceDetails::onOverflowMenuItemEnable(const LLSD& param)
{
	std::string value = param.asString();
	if("can_create_pick" == value)
	{
		return !LLAgentPicksInfo::getInstance()->isPickLimitReached();
	}
	return true;
}

void FSFloaterPlaceDetails::onSLURLBuilt(std::string& slurl)
{
	getWindow()->copyTextToClipboard(utf8str_to_wstring(slurl));
		
	LLSD args;
	args["SLURL"] = slurl;

	LLNotificationsUtil::add("CopySLURL", args);
}

/////////////////////////////////
// Callbacks
/////////////////////////////////

void FSFloaterPlaceDetails::onLandmarkLoaded(LLLandmark* landmark)
{
	if (!mPanelLandmarkInfo)
	{
		return;
	}

	LLUUID region_id;
	landmark->getRegionID(region_id);
	landmark->getGlobalPos(mGlobalPos);
	mPanelLandmarkInfo->displayParcelInfo(region_id, mGlobalPos);
	updateVerbs();
}

void FSFloaterPlaceDetails::changedGlobalPos(const LLVector3d& global_pos)
{
	mGlobalPos = global_pos;
	updateVerbs();
}

void FSFloaterPlaceDetails::processParcelDetails(const LLParcelData& parcel_details)
{
	LLStringUtil::format_map_t args;
	args["[NAME]"] = parcel_details.name.c_str();
	setTitle(getString("title_remote_place_detail", args));
}

void FSFloaterPlaceDetails::changedParcelSelection()
{
	if (!mPanelPlaceInfo)
		return;

	LLViewerParcelMgr* parcel_mgr = LLViewerParcelMgr::getInstance();
	mParcel = parcel_mgr->getFloatingParcelSelection();
	LLParcel* parcel = mParcel->getParcel();
	LLViewerRegion* region = parcel_mgr->getSelectionRegion();
	if (!region || !parcel)
	{
		return;
	}

	LLVector3d prev_pos_global = mGlobalPos;

	// If agent is inside the selected parcel show agent's region<X, Y, Z>,
	// otherwise show region<X, Y, Z> of agent's selection point.
	bool is_current_parcel = fs_is_agent_in_selected_parcel(parcel);
	if (is_current_parcel)
	{
		mGlobalPos = gAgent.getPositionGlobal();
	}
	else
	{
		LLVector3d pos_global = gViewerWindow->getLastPick().mPosGlobal;
		if (!pos_global.isExactlyZero())
		{
			mGlobalPos = pos_global;
		}
	}

	// Reset location info only if global position has changed
	// and update timer has expired to reduce unnecessary text and icons updates.
	if (prev_pos_global != mGlobalPos && mResetInfoTimer.hasExpired())
	{
		mPanelPlaceInfo->resetLocation();
		mResetInfoTimer.setTimerExpirySec(FS_PLACE_INFO_UPDATE_INTERVAL);
	}

	mPanelPlaceInfo->displaySelectedParcelInfo(parcel, region, mGlobalPos, is_current_parcel);

	updateVerbs();
}

void FSFloaterPlaceDetails::updateEstateName(const std::string& name)
{
	if (mPanelPlaceInfo)
	{
		mPanelPlaceInfo->updateEstateName(name);
	}
}

void FSFloaterPlaceDetails::updateEstateOwnerName(const std::string& name)
{
	if (mPanelPlaceInfo)
	{
		mPanelPlaceInfo->updateEstateOwnerName(name);
	}
}

void FSFloaterPlaceDetails::updateCovenantText(const std::string &text)
{
	if (mPanelPlaceInfo)
	{
		mPanelPlaceInfo->updateCovenantText(text);
	}
}
