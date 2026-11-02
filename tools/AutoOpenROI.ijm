// AutoOpenROI.ijm -- ASCII only.
// IMPORTANT: each macro "..." { } block is compiled on its own and CANNOT see
// variables declared at the top of this file. So every path is defined INSIDE
// each block. Only functions are shared.
//
// Install once:  Plugins > Macros > Install...  -> pick this file
// Keys (image window active):
//     n = open NEXT image and load its ROI if present
//     r = reset counter
//     s = save current selection as <image base name>.roi

function baseName(path) {
    nm = File.getName(path);
    dot = lastIndexOf(nm, ".");
    if (dot > 0) return substring(nm, 0, dot);
    return nm;
}

macro "Next Image [n]" {
    imagesDir = "D:/Work/dm_auto_project/agent_work/dataset/images";
    roiDir    = "D:/Work/dm_auto_project/agent_work/dataset/roi";
    idxFile   = "D:/Work/dm_auto_project/agent_work/dataset/tools/_index.txt";
    list = getFileList(imagesDir);
    if (list.length == 0) { print("No images in " + imagesDir); exit; }
    i = 0;
    if (File.exists(idxFile)) {
        s = File.openAsString(idxFile);
        if (s != "") i = parseInt(s);
    }
    if (i >= list.length) { print("All done (" + list.length + "). Press r to restart."); exit; }
    p = imagesDir + "/" + list[i];
    open(p);
    base = baseName(p);
    // close every other open image so only the current one stays on screen
    if (nImages() > 1) {
        titles = getList("image.titles");
        cur = getTitle();
        k = 0;
        while (k < titles.length) {
            if (titles[k] != cur) {
                selectImage(titles[k]);
                close();
                titles = getList("image.titles");
                k = 0;
            } else {
                k++;
            }
        }
        selectImage(cur);
    }
    rp = roiDir + "/" + base + ".roi";
    rz = roiDir + "/" + base + ".zip";
    roiManager("Reset");
    if (File.exists(rp)) {
        roiManager("Open", rp);
        roiManager("Select", 0);
    } else if (File.exists(rz)) {
        roiManager("Open", rz);
        roiManager("Select", 0);
    }
    File.saveString("" + (i + 1), idxFile);
    print("[" + (i + 1) + "/" + list.length + "] " + base);
}

macro "Reset Counter [r]" {
    idxFile = "D:/Work/dm_auto_project/agent_work/dataset/tools/_index.txt";
    File.saveString("0", idxFile);
    print("counter reset to 0 -- press n");
}

macro "Save ROI [s]" {
    roiDir = "D:/Work/dm_auto_project/agent_work/dataset/roi";
    if (selectionType() < 0) {
        rp = roiDir + "/" + baseName(getTitle()) + ".roi";
        if (File.exists(rp)) { File.delete(rp); print("SKIPPED (empty selection) - removed " + baseName(getTitle()) + ".roi"); }
        else { print("SKIPPED (empty selection) - nothing to save"); }
        exit;
    }
    base = baseName(getTitle());
    roiManager("Reset");
    roiManager("Add");
    roiManager("Save", roiDir + "/" + base + ".roi");
    print("saved " + base + ".roi");
}
