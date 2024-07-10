# OCT-SEGMENT-INPAINT CASCADE MODEL (OCTC: Ovarian Cyst Tumor mark detection Cascade model)

## Introduction
The OCTC model is a deep learning-based framework designed to detect and remove identifiable markers from ovarian cyst tumor ultrasound images. This ensures patient privacy while maintaining the diagnostic accuracy of the model. The framework leverages segmentation and inpainting techniques to anonymize ultrasound images effectively.

## Installation
```bash
# Clone the repository
git clone https://github.com/dablro12/octc-cascade.git

# Navigate to the project directory
cd octc-cascaded

# Install the required dependencies
pip install -r env/requirements.txt
```

## Usage
```bash
# To visualize the results
chmod +x run.sh
bash run.sh
```

## Results
The model demonstrates high performance in both segmentation and inpainting tasks, achieving a Dice score of 0.934 and an accuracy of 0.953 for the segmentation model (SwinUNETRv2). The inpainting model (OCI-GAN) achieves an MSE of 0.0076, ensuring high-quality reconstruction of the anonymized ultrasound images.

## References
- Clunie, David A., et al. "Report of the Medical Image De-Identification (MIDI) Task Group-Best Practices and Recommendations." Arxiv (2023).
- Chen, Lijiang, et al. "Improving the segmentation accuracy of ovarian-tumor ultrasound images using image inpainting." Bioengineering 10.2 (2023): 184.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgement
We would like to thank the Department of Transdisciplinary Medicine, Innovative Medical Technology Research Institute, and the Department of Medicine at Seoul National University for their support and collaboration.

## Contact
For any questions or inquiries, please contact:

- Daehyeon Choe: dablro1232@gmail.com
- Seoi Jeong: selee203@snu.ac.kr
- Hyoun Joong Kong: gongcop7@snu.ac.kr

## Citation
If you use this work in your research, please cite:

```bibtex
@article{choe2024octc,
  title={OCT-SEGMENT-INPAINT CASCADE MODEL for Ovarian Cyst Tumor mark detection},
  author={Choe, Daehyeon and Jeong, Seoi and Kong, Hyoun Joong},
  journal={Proceedings of KSCI Conference},
  year={2024}
}
```

## Author
- Daehyeon Choe
- Seoi Jeong
- Hyoun Joong Kong

## Version
Current version: 1.0.0
