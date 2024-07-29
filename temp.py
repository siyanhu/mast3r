import sys
sys.path.append('/home/siyanhu/Gits/mast3r')

import root_file_io as fio

if __name__ == '__main__':
    db_tag = '7scenes'
    data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'data', db_tag])
    scene_dirs = fio.traverse_dir(data_dir, full_path=True, towards_sub=False)
    scene_dirs = fio.filter_if_dir(scene_dirs, filter_out_target=False)

    for scn_dth in scene_dirs:
        (scndir, scnname, scnext) = fio.get_filename_components(scn_dth)
        seq_dirs = fio.traverse_dir(scn_dth, full_path=True, towards_sub=False)
        seq_dirs = fio.filter_if_dir(seq_dirs, filter_out_target=False)

        for seq_dth in seq_dirs:
            full_pths = fio.traverse_dir(seq_dth, full_path=True, towards_sub=False)
            img_pths = fio.filter_folder(full_pths, filter_out=False, filter_text='color')
            img_pths = fio.filter_ext(img_pths, filter_out_target=False, ext_set=fio.img_ext_set)

            for file_pth in full_pths:
                if file_pth not in img_pths:
                    fio.delete_file(file_pth)

        