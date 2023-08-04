import os

def rename_files(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename == 'tensor_B_mode_0_crd':
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, 'tensor_B_mode_1_crd1')
                os.rename(old_path, new_path)
            if filename == 'tensor_B_mode_0_seg':
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, 'tensor_B_mode_1_seg1')
                os.rename(old_path, new_path)

            # if filename == 'tensor_B_mode_1_crd':
            #     old_path = os.path.join(root, filename)
            #     new_path = os.path.join(root, 'tensor_B_mode_0_crd')
            #     os.rename(old_path, new_path)
            # if filename == 'tensor_B_mode_1_seg':
            #     old_path = os.path.join(root, filename)
            #     new_path = os.path.join(root, 'tensor_B_mode_0_seg')
            #     os.rename(old_path, new_path)

            # if filename == 'tensor_B_mode_1_crd1':
            #     old_path = os.path.join(root, filename)
            #     new_path = os.path.join(root, 'tensor_B_mode_1_crd')
            #     os.rename(old_path, new_path)
            # if filename == 'tensor_B_mode_1_seg1':
            #     old_path = os.path.join(root, filename)
            #     new_path = os.path.join(root, 'tensor_B_mode_1_seg')
            #     os.rename(old_path, new_path)

                # print(f'Renamed "{filename}" to "file0" in {root}')

if __name__ == "__main__":
    target_directory = "/nobackup/jadivara/sam/SUITESPARSE_FORMATTED"  # Replace this with the actual directory path
    rename_files(target_directory)