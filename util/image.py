import os
import errno
from datetime import datetime


def save_image(root_dir, file):
    """
    Save image
    :param root_dir: root dir of all images
    :param file: python file object
    :return:
    """
    t_date = datetime.now()
    img_dir = os.path.join(root_dir, t_date.strftime("%Y-%m-%d"))
    create_dir_if_not_exist(img_dir)
    file_path = os.path.join(img_dir, t_date.strftime("%H-%M-%S")+"-"+file.filename)
    file.save(file_path)
    return file_path


def create_dir_if_not_exist(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

