/** 
 * @file llpanelmaininventory.h
 * @brief llpanelmaininventory.h
 * class definition
 *
 * $LicenseInfo:firstyear=2001&license=viewerlgpl$
 * Second Life Viewer Source Code
 * Copyright (C) 2010, Linden Research, Inc.
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

#ifndef LL_LLPANELMAININVENTORY_H
#define LL_LLPANELMAININVENTORY_H

#include "llpanel.h"
#include "llinventoryfilter.h"
#include "llinventoryobserver.h"
#include "lldndbutton.h"

#include "llfolderview.h"

class LLComboBox;
class LLFolderViewItem;
class LLInventoryPanel;
class LLSaveFolderState;
class LLFilterEditor;
class LLTabContainer;
class LLFloaterInventoryFinder;
class LLMenuButton;
class LLMenuGL;
class LLToggleableMenu;
class LLFloater;
class LLComboBox;	// <FS:Zi> Filter dropdown

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Class LLPanelMainInventory
//
// This is a panel used to view and control an agent's inventory,
// including all the fixin's (e.g. AllItems/RecentItems tabs, filter floaters).
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class LLPanelMainInventory : public LLPanel, LLInventoryObserver
{
public:
	friend class LLFloaterInventoryFinder;

	LLPanelMainInventory(const LLPanel::Params& p = getDefaultParams());
	~LLPanelMainInventory();

	BOOL postBuild();

	virtual BOOL handleKeyHere(KEY key, MASK mask);

	// Inherited functionality
	/*virtual*/ BOOL handleDragAndDrop(S32 x, S32 y, MASK mask, BOOL drop,
									   EDragAndDropType cargo_type,
									   void* cargo_data,
									   EAcceptance* accept,
									   std::string& tooltip_msg);
	/*virtual*/ void changed(U32);
	/*virtual*/ void draw();
	/*virtual*/ void 	onVisibilityChange ( BOOL new_visibility );
	// <FS:Ansariel> CTRL-F focusses local search editor
	/*virtual*/ bool hasAccelerators() const { return true; }

	LLInventoryPanel* getPanel() { return mActivePanel; }
	LLInventoryPanel* getActivePanel() { return mActivePanel; }
	const LLInventoryPanel* getActivePanel() const { return mActivePanel; }
	LLInventoryPanel* getAllItemsPanel();
	void selectAllItemsPanel();
	// <FS:Ansariel> FIRE-19493: "Show Original" should open main inventory panel
	void showAllItemsPanel();
	// </FS:Ansariel>

	bool isRecentItemsPanelSelected();

	const std::string& getFilterText() const { return mFilterText; }
	
	void setSelectCallback(const LLFolderView::signal_t::slot_type& cb);

	void onFilterEdit(const std::string& search_string );

	void setFocusFilterEditor();

	static void newWindow();

	void toggleFindOptions();

    void resetFilters();

	// <FS:Zi> Filter dropdown
	void onFilterTypeSelected(const std::string& filter_type_name);
	void updateFilterDropdown(const LLInventoryFilter* filter);
	// </FS:Zi> Filter dropdown

	void doCustomAction(const LLSD& userdata) { onCustomAction(userdata); } // <FS:Ansariel> Prevent warning "No callback found for: 'Inventory.CustomAction' in control: Find Links"

	// <FS:Ansariel> FIRE-12808: Don't save filters during settings restore
	static bool sSaveFilters;

protected:
	//
	// Misc functions
	//
	void setFilterTextFromFilter();
	void startSearch();
	
	void onSelectionChange(LLInventoryPanel *panel, const std::deque<LLFolderViewItem*>& items, BOOL user_action);

	static BOOL filtersVisible(void* user_data);
	void onClearSearch();
	static void onFoldersByName(void *user_data);
	static BOOL checkFoldersByName(void *user_data);
	
	static BOOL incrementalFind(LLFolderViewItem* first_item, const char *find_text, BOOL backward);
	void onFilterSelected();

	const std::string getFilterSubString();
	void setFilterSubString(const std::string& string);

	// menu callbacks
	void doToSelected(const LLSD& userdata);
	void closeAllFolders();
	void doCreate(const LLSD& userdata);

	// <FS:Zi> Sort By menu handlers
	void setSortBy(const LLSD& userdata);
	BOOL isSortByChecked(const LLSD& userdata);
	// </FS:Zi> Sort By menu handlers

	void saveTexture(const LLSD& userdata);
	bool isSaveTextureEnabled(const LLSD& userdata);
	void updateItemcountText();

	// <FS:Zi> Inventory Collapse and Expand Buttons
	void onCollapseButtonClicked();
	void onExpandButtonClicked();
	// </FS:Zi> Inventory Collapse and Expand Buttons
	void onFocusReceived();
	void onSelectSearchType();
	void updateSearchTypeCombo();

private:
	LLFloaterInventoryFinder* getFinder();

	LLFilterEditor*				mFilterEditor;
	LLTabContainer*				mFilterTabs;
	LLUICtrl*					mCounterCtrl;
	LLHandle<LLFloater>			mFinderHandle;
	LLInventoryPanel*			mActivePanel;
	LLInventoryPanel*			mWornItemsPanel;
	bool						mResortActivePanel;
	LLSaveFolderState*			mSavedFolderState;
	std::string					mFilterText;
	std::string					mFilterSubString;
	S32							mItemCount;
	std::string					mItemCountString;
	S32							mCategoryCount;
	std::string					mCategoryCountString;
	LLComboBox*					mSearchTypeCombo;

	// <FS:Zi> Filter dropdown
	LLComboBox*					mFilterComboBox;
	std::map<std::string,U64>	mFilterMap;			// contains name-to-number mapping for dropdown filter types
	U64							mFilterMask;		// contains the cumulated bit filter for all dropdown filter types
	// </FS:Zi> Filter dropdown


	//////////////////////////////////////////////////////////////////////////////////
	// List Commands                                                                //
protected:
	void initListCommandsHandlers();
	void updateListCommands();
	void onAddButtonClick();
	void showActionMenu(LLMenuGL* menu, std::string spawning_view_name);
	void onTrashButtonClick();
	void onClipboardAction(const LLSD& userdata);
	BOOL isActionEnabled(const LLSD& command_name);
	BOOL isActionChecked(const LLSD& userdata);
	void onCustomAction(const LLSD& command_name);

	// <FS:Zi> FIRE-31369: Add inventory filter for coalesced objects
	void onCoalescedObjectsToggled(const LLSD& userdata);
	bool isCoalescedObjectsChecked(const LLSD& userdata);
	// </FS:Zi>

	// <FS:Zi> Filter Links Menu
	BOOL isFilterLinksChecked(const LLSD& userdata);
	void onFilterLinksChecked(const LLSD& userdata);
	// </FS:Zi> Filter Links Menu

	// <FS:Zi> FIRE-1175 - Filter Permissions Menu
	BOOL isFilterPermissionsChecked(const LLSD &userdata);
	void onFilterPermissionsChecked(const LLSD &userdata);
	// </FS:Zi>

	// <FS:Zi> Extended Inventory Search
	BOOL isSearchTypeChecked(const LLSD& userdata);
	void onSearchTypeChecked(const LLSD& userdata);
	// </FS:Zi> Extended Inventory Search

	bool handleDragAndDropToTrash(BOOL drop, EDragAndDropType cargo_type, EAcceptance* accept);
    static bool hasSettingsInventory();
	/**
	 * Set upload cost in "Upload" sub menu.
	 */
	void setUploadCostIfNeeded();
private:
	LLDragAndDropButton*		mTrashButton;
	LLToggleableMenu*			mMenuGearDefault;
	LLToggleableMenu*			mMenuVisibility;
	LLMenuButton*				mGearMenuButton;
	LLMenuButton*				mVisibilityMenuButton;
	LLHandle<LLView>			mMenuAddHandle;

	// <FS:Zi> Inventory Collapse and Expand Buttons
	LLButton*					mCollapseBtn;
	LLButton*					mExpandBtn;
	// </FS:Zi> Inventory Collapse and Expand Buttons

	bool						mNeedUploadCost;
	// List Commands                                                              //
	////////////////////////////////////////////////////////////////////////////////
};

#endif // LL_LLPANELMAININVENTORY_H



