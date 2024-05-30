

class Seg_Inpaint_Service:
    def __init__(self, img_dir, mask_dir, save_path, visual_save_path, model_name, model_path, sql_config_path, batch_size=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.save_dir = save_path
        self.visual_save_dir = visual_save_path  # 이 경로를 올바르게 설정해야 합니다.
        self.model_name=  model_name
        self.model_path = model_path
        self.sql_config_path = sql_config_path
        # 경로가 디렉토리인지 확인하고, 필요하다면 생성합니다.
        # os.makedirs(self.save_dir, exist_ok=True)
        # os.makedirs(self.visual_save_dir, exist_ok=True)  # 오류가 발생하는 부분, 올바른 디렉토리 경로인지 확인해야 합니다.
        '''
        # #mysql의 정보를 받아오는 코드입니다.
        # with open(self.sql_config_path, 'r') as f:
        #     mysql_config = json.load(f)

        self.host = mysql_config['host']
        self.user = mysql_config['user']
        self.password = mysql_config['password']
        self.database = mysql_config['database']
        '''
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.load_model() # weight model 설정 및 로드
        self.setup_dataloader()        # 데이터 로더 설정

    def setup_dataloader(self):
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        test_dataset = CustomDataset(
            image_dir = self.img_dir,
            mask_dir = self.mask_dir,
            transform= test_transform,
            mask_transform= test_transform,
            testing = True,
            mask_shuffle = False,
        )
        self.test_loader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, shuffle = False)



    def load_model(self):
        if self.model_name == 'ocigan':
            from model.aotgan import InpaintGenerator
            self.netG = InpaintGenerator().to(self.device)
        elif self.model_name == 'vae':
            from model.vae import Encoder, VQEmbeddingEMA, Decoder, Model
            input_dim = 4
            hidden_dim =512
            latent_dim = 32
            n_embeddings = 512
            output_dim = 3
            encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=latent_dim)
            codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=latent_dim)
            decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            self.netG = Model(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(self.device)

        else:
            raise ValueError("#"* 30 , "Invalid model name", "#"* 30)
        model_weights = torch.load(self.model_path, map_location=self.device)['netG_state_dict']
        self.netG.load_state_dict(model_weights)
        self.netG.eval()

    def save_result(self, input_images, masks, comp_images, image_paths):
        try:
            # MySQL 데이터베이스 연결
            conn = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )

            cursor = conn.cursor()

            for input_image, mask, pred_image, path in zip(input_images, masks, comp_images, image_paths):
                file_name = self.model_name + '_' + path.split('/')[-1]
                # Origianl Image 대로 Resize
                original_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                utils.test_plotting(input_image, mask, pred_image, save_path=os.path.join(self.visual_save_dir, file_name))

                pred_image = pred_image.cpu().detach().numpy()
                # Transpose pred_image from (C, H, W) to (H, W, C) for OpenCV compatibility
                pred_image = np.transpose(pred_image, (1, 2, 0))

                # Now, resize pred_image to match the original image's dimensions
                pred_image = cv2.resize(pred_image, (original_image.shape[1], original_image.shape[0]))  # Note the order of shape
                pred_image = np.clip(pred_image * 255, 0, 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
                cv2.imwrite(os.path.join(self.save_dir, file_name), pred_image)

                # 결과를 MySQL 데이터베이스에 저장
                query = "UPDATE IMAGEBOARD SET image2 = %s WHERE image = %s"
                values = ('/inpainted/'+ file_name, '/image/'+ file_name)
                cursor.execute(query, values)
                conn.commit()

            print("쿼리 실행 완료")
            print(os.path.join(self.save_dir, file_name))

        except pymysql.Error as e:
            print("MySQL 에러 발생:", e)

        finally:
            # 연결 닫기
            if conn:
                conn.close() 
    def infer(self):
        with torch.no_grad():
            for images, masks, image_paths in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                ### 전처리
                masks = utils.infer_preprocess_mask(images, masks)

                # mask가 0이 아닌 부분에 대해 image를 mask로 대체
                input_images = images.clone()
                # mask와 input_images shape이 같아야하므로 mask를 image shape으로 resize
                input_images[masks != 0] = masks[masks != 0] 
                # input_images 처리해줫으니 다시 masks를 1채널로 변경
                masks = masks[:,0,:,:].unsqueeze(1)
                # 입력이미지 device 할당
                input_images = input_images.to(self.device) 

                # inference of different models
                if self.model_name == 'ocigan':
                    pred_images = self.oci_gan_inference(input_images, masks)
                elif self.model_name == 'vae':
                    pred_images =  self.vae_inference(input_images, masks)
                        
                ## mask에서 0이 아닌 부분을 GT로 대체, 이때 마스크는 0~1사이의 값을 가짐 
                comp_images = images.clone()
                comp_images[masks.repeat(1,3,1,1) != 0] = pred_images[masks.repeat(1,3,1,1) != 0] #comp_image를 남겨두는 이유는 result를 확인하기 위함 
                self.save_result(input_images, masks, comp_images, image_paths)

    def oci_gan_inference(self, input_images, masks):
        pred_images = self.netG(input_images, masks)  # 3+1ch
        ## mask에서 0이 아닌 부분을 GT로 대체, 이때 마스크는 0~1사이의 값을 가짐 
        return pred_images
    def vae_inference(self, input_images, masks):
        ### inference
        pred_images, _ = self.netG(input_images, masks)
        ## mask에서 0이 아닌 부분을 GT로 대체, 이때 마스크는 0~1사이의 값을 가짐 
        return pred_images
        