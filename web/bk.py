import streamlit as st
import streamlit_authenticator as stauth
import yaml
from PIL import Image
import os
import subprocess

class ImageProcessingApp:
    def __init__(self, input_dir, output_dir, script_path, config_path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.script_path = script_path
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 스크립트 경로가 정확한지 확인하고, 실행 권한을 부여합니다.
        if os.path.exists(self.script_path):
            if not os.access(self.script_path, os.X_OK):
                os.chmod(self.script_path, 0o775)
        else:
            st.error(f"스크립트 경로를 찾을 수 없습니다: {self.script_path}")
    
        self.authenticator = self.init_privacy(config_path)
        
    def init_privacy(self, config_path):
        with open(config_path) as file:
            config = yaml.load(file, Loader=stauth.SafeLoader)
            
        # yaml 파일 데이터로 객체 생성
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized']
        )
        
        return authenticator

    def upload_image(self):
        st.header("1. 이미지 입력")
        self.uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
        if self.uploaded_file is not None:
            self.image = Image.open(self.uploaded_file)
            st.image(self.image, caption="업로드한 이미지", use_column_width=True)
            self.save_image(self.image, self.input_dir, self.uploaded_file.name)
            st.write(f"이미지가 '{os.path.join(self.input_dir, self.uploaded_file.name)}'에 저장되었습니다.")
            return True
        else:
            return False
    
    def save_image(self, image, directory, filename):
        path = os.path.join(directory, filename)
        image.save(path)
    
    def process_image(self):
        st.header("2. 이미지 처리")
        process_button = st.button("이미지 처리 실행")
        if process_button:
            image_path = os.path.join(self.input_dir, self.uploaded_file.name)
            self.run_shell_script(image_path)
            output_image_path = os.path.join(self.output_dir, self.uploaded_file.name)
            try:
                if os.path.exists(output_image_path):
                    processed_image = Image.open(output_image_path)
                    cols = st.columns(2)
                    cols[0].image(self.image, caption="처리 전 이미지", use_column_width=True)
                    cols[1].image(processed_image, caption="처리 후 이미지", use_column_width=True)
                    
                    # 각 이미지에 다운로드 버튼 추가
                    with open(output_image_path, "rb") as file:
                        cols[1].download_button(
                            label="처리된 이미지 다운로드",
                            data=file,
                            file_name=f"processed_{self.uploaded_file.name}",
                            mime="image/png"
                        )
                    
                    st.write(f"처리된 이미지가 '{output_image_path}'에 저장되었습니다.")
                else:
                    st.write("처리된 이미지를 불러올 수 없습니다. 스크립트를 확인해주세요.")
            except Exception as e:
                st.write(f"처리된 이미지를 불러오는 중 오류가 발생했습니다: {e}")
    
    def run_shell_script(self, image_path):
        try:
            # 쉘 스크립트를 실행합니다.
            result = subprocess.run(["sh", self.script_path, image_path], capture_output=True, text=True)
            if result.returncode == 0:
                st.write("스크립트가 성공적으로 실행되었습니다.")
            else:
                st.write("스크립트 실행 중 오류가 발생했습니다.")
                st.write(result.stderr)
        except Exception as e:
            st.write(f"스크립트 실행 중 예외가 발생했습니다: {e}")
    
    def display_saved_images(self):
        st.header("저장된 이미지 보기")
        saved_images = os.listdir(self.output_dir)
        if saved_images:
            for img_name in saved_images:
                cols = st.columns(2)
                # 1열에는 원본이미지, 2열에는 처리된 이미지를 출력
                input_img_path = os.path.join(self.input_dir, img_name)
                output_img_path = os.path.join(self.output_dir, img_name)
                
                # 원본 이미지는 1열 출력
                cols[0].image(input_img_path, caption=f"원본 이미지 ({img_name})", use_column_width=True)
                # 처리된 이미지는 2열 출력
                cols[1].image(output_img_path, caption=f"처리된 이미지 ({img_name})", use_column_width=True)
                
                # 각 원본 이미지에 다운로드 버튼 추가
                with open(input_img_path, "rb") as file:
                    cols[0].download_button(
                        label="원본 이미지 다운로드",
                        data=file.read(),
                        file_name=f"original_{img_name}",
                        mime="image/png"
                    )
                
                # 각 처리된 이미지에 다운로드 버튼 추가
                with open(output_img_path, "rb") as file:
                    cols[1].download_button(
                        label="처리된 이미지 다운로드",
                        data=file.read(),
                        file_name=f"processed_{img_name}",
                        mime="image/png"
                    )
        else:
            st.write("저장된 이미지가 없습니다.")
    
    def run(self):
        name, authentication_status, username = self.authenticator.login("Login", "main")

        if authentication_status:
            self.authenticator.logout("Logout", "sidebar")
            st.sidebar.title(f"Welcome {name}")

            st.title("[DCS2024] Ultrasound Marker Auto-Inpainting Service using Cascade Seg-Inpaint DeepLearning Model")

            page = st.sidebar.selectbox("페이지를 선택하세요", ["프로젝트 소개", "서비스 이용"])

            if page == "프로젝트 소개":
                st.markdown("""
                ## Introduction 
                - **Paper Link** : Auto Ultrasound Marker Inpainting Model : Cascade Seg-Inpaint DeepLearning Model
                - **Authors** : Dae hyeon Choe1, *, Seoi Jeong1, and Hyoun Joong Kong1, 2, 3, †
                - **Affiliations** :
                1 Transdisciplinary Department of Medicine, Seoul National University Hospital, Seoul 03122, Republic of Korea
                2 Interdisciplinary Program in Bioengineering, Graduate School, Seoul National University, Seoul 08826, Republic of Korea
                3 Department of Biomedical Engineering, Seoul National University College of Medicine, Seoul 03080, Republic of Korea
                
                ## Abstract
                    이 프로젝트는 이미지 분할 및 인페인팅을 위한 웹 애플리케이션입니다. 사용자는 이미지를 업로드하고, 
                    딥러닝 모델을 사용하여 이미지를 처리한 후 결과를 확인할 수 있습니다.
                    
                ## Usages 
                1. **이미지 업로드**: 왼쪽의 파일 업로드 버튼을 사용하여 이미지를 업로드합니다.
                2. **이미지 처리**: '이미지 처리 실행' 버튼을 눌러 업로드한 이미지를 처리합니다.
                3. **결과 확인 및 다운로드**: 처리된 이미지를 확인하고 다운로드 버튼을 눌러 이미지를 저장할 수 있습니다.
                
                ## Main Function
                - 이미지 업로드
                - 이미지 처리 (분할 및 인페인팅)
                - 처리된 이미지 다운로드
                
                ## Flow Chart
                """)

                # 이미지 추가
                flow_chart1_path = "/home/eiden/eiden/octc-cascade/web/ui/segmentation-over-flow.png"
                flow_chart2_path = "/home/eiden/eiden/octc-cascade/web/ui/inpaint-over-flow.png"
                # 2열로 이미지 표시
                cols = st.columns(2)
                cols[0].image(flow_chart1_path, caption="Flow Chart", use_column_width=True)
                cols[1].image(flow_chart2_path, caption="Flow Chart", use_column_width=True)
                
                st.markdown("""
                ## Model Architecture
                """)

                # 이미지 추가
                architecture1_path = "/home/eiden/eiden/octc-cascade/web/ui/oci-gan.png"
                architecture2_path = "/home/eiden/eiden/octc-cascade/web/ui/oci-gan.png"
                # 2열로 이미지 표시
                cols = st.columns(2)
                cols[0].image(architecture1_path, caption="Model Architecture", use_column_width=True)
                cols[1].image(architecture2_path, caption="Model Architecture", use_column_width=True)
                
                st.markdown("""
                ## 프로젝트 정보
                - GitHub: [octc-cascade](https://github.com/dablro12/octc-cascade)
                - Web Supervisor : Daehyeon Choe
                - Email : dablro1232@gmail.com
                """)

            elif page == "서비스 이용":
                if self.upload_image():
                    self.process_image()
                self.display_saved_images()

        elif authentication_status == False:
            st.error("Username/password is incorrect")
        elif authentication_status == None:
            st.warning("Please enter your username and password")

if __name__ == "__main__":
    project_dir = '/home/eiden/eiden/octc-cascade'
    app = ImageProcessingApp(
        input_dir=os.path.join(project_dir, "web/DB/data"),
        output_dir=os.path.join(project_dir, "web/DB/output"),
        script_path=os.path.join(project_dir, 'inference/script/run.sh'),
        config_path=os.path.join(project_dir, "web/settings/config.yaml")
    )
    app.run()
