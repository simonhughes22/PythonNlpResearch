import os
import shutil
from FindFiles import find_files
import time

def move(from_dir, to_dir, regex, ignore_empty = True, replace_existing = False):
    lst_files = find_files(from_dir, regex, remove_empty = not ignore_empty)
    cnt = 0
    for f in lst_files:
        dfile = os.path.join(to_dir, os.path.basename(f))
        if os.path.exists(dfile):
            if not replace_existing:
                print "%s already exists" % dfile
            else:
                os.remove(dfile)
        else:

            shutil.copyfile(f, dfile)
            cnt += 1
    print "Moved %s files" % str(cnt)

def replace_if_newer(from_dir, to_dir, regex, ignore_empty = True):
    lst_files = find_files(from_dir, regex, remove_empty = ignore_empty)
    cnt = 0
    for from_file in lst_files:
        to_file = os.path.join(to_dir, os.path.basename(from_file))
        if os.path.exists(to_file):
            from_time = time.ctime(os.path.getmtime(from_file))
            to_time = time.ctime(os.path.getmtime(to_file))

            if from_time >= to_time:
                os.remove(to_file)
                shutil.copyfile(from_file, to_file)
            else:
                print "Passing on %s as destination file is newer" % from_file
        else:

            shutil.copyfile(from_file, to_file)
            cnt += 1
    print "Moved %s files" % str(cnt)

