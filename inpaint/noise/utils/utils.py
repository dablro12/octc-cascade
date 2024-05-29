import os 
import cv2
def find_files(directory, extension):
    """
    특정 경로에서 특정 확장자를 가진 파일을 찾는 함수
    :param directory: 파일을 찾을 경로
    :param extension: 찾을 파일의 확장자
    """
    file_list = []
    # 디렉토리 내의 모든 파일 및 디렉토리를 검사
    for dir in os.listdir(directory):
        if not dir.endswith('csv'):
            for label in os.listdir(os.path.join(directory, dir)):
                for file in os.listdir(os.path.join(directory, dir, label)):
                    # 파일의 확장자가 일치하는 경우
                    if file.endswith(extension):
                        # 파일의 절대 경로를 리스트에 추가
                        file_list.append(os.path.join(directory, dir, label, file).replace('\\', '/'))
                
    return file_list

def save_files(load_files):
    """
    Binary Thresholding 된 이미지를 저장하는 함수

    Args:
        load_files (_li_): 원본 파일 경로로 구성된 리스트
        extension (_str_): ".png"
    return:
        None
    """
    for file in load_files:
        cv2.imwrite(file.replace('origin', 'binary_aot_mask_v2'), file)
        # print(file.replace('sono', 'binary_mask'))
        