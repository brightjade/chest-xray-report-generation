# Towards Accurate Biomedical Report Generation for Chest X-Ray Radiographs

![fig2_edited-1](https://user-images.githubusercontent.com/33290571/153551192-6ddf649a-8a56-4626-b1e3-96715a9c5698.png)

*This project was done while taking an AI604 course (Computer Vision) at KAIST.*

## Project Summary

With the overwhelming demand of human work in the medical field, the automation to speed up parts of the medical treatment process has become a critical issue. Specifically, the radiology report writing of chest X-ray radiographs requires trained specialists in order to analyze and detect abnormalities in the image; however, experienced workers are rare in the field, which calls for automating the report generation process. Prior studies have attempted to generate reports in a hierarchical manner, decoding one sentence at a time, but such approach is inefficient in computation time, let alone generates high-quality reports. In this work, we propose a chest X-ray radiology report generation framework based on Transformer that aims to create a paragraph-level report in a single pass given a chest X-ray radiograph. In order to generate not just realistic but also accurate reports, we train our network in a multi-task learning fashion, also accounting for the accuracy of the generated reports. We evaluate our framework on a publicly available chest X-ray report dataset and demonstrate that our model is comparable to existing models in terms of both clinical accuracy and natural language generation metrics.
