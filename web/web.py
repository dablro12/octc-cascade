from app.image_loader import * 
from settings.auth import Authenticator
class ImageProcessingApp:
    def __init__(self, input_dir, output_dir, script_path, config_path):
        self.authenticator = Authenticator(config_path)
        self.uploader = ImageUploader(input_dir)
        self.processor = ImageProcessor(script_path)
        self.viewer = ImageViewer(output_dir)
    
    def run(self):
        name, authentication_status, username = self.authenticator.login()

        if authentication_status:
            self.authenticator.logout(name)
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
                filename, image = self.uploader.upload_image()
                if filename:
                    output_image_path = self.processor.process_image(self.uploader.input_dir, filename)
                    if output_image_path:
                        print(username)
                        self.viewer.display_saved_images(username)
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
