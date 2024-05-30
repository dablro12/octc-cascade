from PIL import Image
import subprocess
import streamlit as st
import os

class ImageUploader:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        os.makedirs(self.input_dir, exist_ok=True)
    
    def upload_image(self):
        st.header("1. 이미지 입력")
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드한 이미지", use_column_width=True)
            self.save_image(image, self.input_dir, uploaded_file.name)
            st.write(f"이미지가 선택되었습니다.")
            return uploaded_file.name, image
        else:
            return None, None
    
    def save_image(self, image, directory, filename):
        path = os.path.join(directory, filename)
        image.save(path)



class ImageProcessor:
    def __init__(self, script_path):
        self.script_path = script_path
        if os.path.exists(self.script_path):
            if not os.access(self.script_path, os.X_OK):
                os.chmod(self.script_path, 0o775)
        else:
            st.error(f"스크립트 경로를 찾을 수 없습니다: {self.script_path}")
    
    def process_image(self, input_dir, filename):
        st.header("2. Auto-Inpainting Process 처리하기")
        process_button = st.button("Auto-Inpainting")
        if process_button:
            image_path = os.path.join(input_dir, filename)
            self.run_shell_script(image_path)
            output_image_path = os.path.join(input_dir.replace('data', 'output'), filename)
            return output_image_path
        else:
            return None
    
    def run_shell_script(self, image_path):
        try:
            result = subprocess.run(["sh", self.script_path, image_path], capture_output=True, text=True)
            if result.returncode == 0:
                st.write("스크립트가 성공적으로 실행되었습니다.")
            else:
                st.write("스크립트 실행 중 오류가 발생했습니다.")
                st.write(result.stderr)
        except Exception as e:
            st.write(f"스크립트 실행 중 예외가 발생했습니다: {e}")


class ImageViewer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def display_saved_images(self, username):
        st.header("저장된 이미지 보기")
        saved_images = os.listdir(self.output_dir)
        if saved_images:
            for img_name in saved_images:
                output_img_path = os.path.join(self.output_dir, img_name)
                cols = st.columns(2)

                if username == "dablro12":
                    input_img_path = os.path.join(self.output_dir.replace('output', 'data'), img_name)
                    cols[0].image(input_img_path, caption=f"원본 이미지 ({img_name})", use_column_width=True)
                    
                    with open(input_img_path, "rb") as file:
                        cols[0].download_button(
                            label="원본 이미지 다운로드",
                            data=file.read(),
                            file_name=f"original_{img_name}",
                            mime="image/png"
                        )

                cols[1].image(output_img_path, caption=f"처리된 이미지 ({img_name})", use_column_width=True)
                
                with open(output_img_path, "rb") as file:
                    cols[1].download_button(
                        label="처리된 이미지 다운로드",
                        data=file.read(),
                        file_name=f"processed_{img_name}",
                        mime="image/png"
                    )
                
                
        else:
            st.write("저장된 이미지가 없습니다.")
