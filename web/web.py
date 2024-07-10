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
            st.title("[KSCI2024] 개인정보 보호를 위한 초음파 영상 마커 제거 및 복원 모델(PPUIDM): 딥러닝 기반 익명화 웹 서비스")

            page = st.sidebar.selectbox("페이지를 선택하세요", ["프로젝트 소개", "서비스 이용"])

            if page == "프로젝트 소개":
                st.markdown("""
                ## Introduction 
                - **Paper Link** : 개인정보 보호를 위한 초음파 영상 마커 제거 및 복원 모델(PPUIDM) 딥러닝 기반 익명화 프레임워크
                - **Authors** : Dae hyeon Choe1, *, Seoi Jeong1, and Hyoun Joong Kong1, 2, 3, †
                - **Affiliations** 
                
                서울대학교병원 융합의학과
                
                서울대학교 바이오엔지니어링 전공 
                
                서울대학교 의과대학 바이오메디컬엔지니어링 전공 
                - **Contact** : dablro1232@gmail.com
                
                - **Keyword**
                Segmentation, Inpainting, GAN, De-marker Model, Medical Imaging Processing
                
                ## **Abstract**
                본 연구는 초음파 영상에서 환자의 개인 정보를 보호하기 위해 마커를 제거하고 복원하는 딥러닝 기반 자동화 프레임워크(PPUIDM)를 제안하고 그 효과를 평가하였다. 이 프레임워크는 초음파 영상의 환자 정보를 제거하여 프라이버시를 보호하고 딥러닝 모델의 진단 정확도를 향상시키기 위해 개발되었다. SwinUNETRv2 모델을 사용하여 마커와 주석을 식별하고, OCI-GAN 모델을 통해 자연스러운 텍스처로 복원하였다. 실험 결과, Inpainting Model의 경우 OCI-GAN 모델이 MSE 0.0076을 기록하며 우수한 성능을 보였다. 결론적으로, 본 연구는 딥러닝 모델의 보조진단 신뢰성을 높이고, 연구 및 교육적 목적으로 개인정보 보호 문제를 최소화하는 기술을 제공한다.
                
                ## **Usages** 
                1. **이미지 업로드**: 왼쪽의 파일 업로드 버튼을 사용하여 이미지를 업로드합니다.
                2. **이미지 처리**: '이미지 처리 실행' 버튼을 눌러 업로드한 이미지를 처리합니다.
                3. **결과 확인 및 다운로드**: 처리된 이미지를 확인하고 다운로드 버튼을 눌러 이미지를 저장할 수 있습니다.
                
                ## Main Function
                - 이미지 업로드
                - 이미지 처리 (분할 및 인페인팅)
                - 처리된 이미지 다운로드
                
                ## Data/DL Flow Chart
                """)

                # 이미지 추가
                flow_chart1_path = "/home/eiden/eiden/octc-cascade/web/ui/data_flow.png"
                flow_chart2_path = "/home/eiden/eiden/octc-cascade/web/ui/overall.png"
                # 2열로 이미지 표시
                cols = st.columns(2)
                cols[0].image(flow_chart1_path, caption="Data Flow Chart", use_column_width=True)
                cols[1].image(flow_chart2_path, caption="DL Flow Chart", use_column_width=True)
                
                st.markdown("""
                ## DL Process Algorithm
                """)
                # 이미지 추가
                architecture1_path = "/home/eiden/eiden/octc-cascade/web/ui/algorithm.png"
                # 1열로 이미지 표시
                cols = st.columns(1)
                cols[0].image(architecture1_path, caption="Model Process Algorithm", use_column_width=True)

                st.markdown("""
                ## PPUIDM Result
                """)

                # 이미지 추가
                architecture1_path = "/home/eiden/eiden/octc-cascade/web/ui/input.png"
                architecture2_path = "/home/eiden/eiden/octc-cascade/web/ui/output.png"
                # 2열로 이미지 표시
                cols = st.columns(2)
                cols[0].image(architecture1_path, caption="Original Data", use_column_width=True)
                cols[1].image(architecture2_path, caption="Clean Data(Result)", use_column_width=True)
                
                
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
