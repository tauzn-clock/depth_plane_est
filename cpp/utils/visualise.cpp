#include <opencv2/opencv.hpp>

std::array<int,3> hsv(int a, int n){
    int h = (int)(a*360/n);
    int s = 255;
    int v = 255;

    // Convert to RGB
    int r, g, b;
    int i = (int)(h / 60) % 6;
    int f = (int)(h % 60);
    int p = (int)(v * (1 - s / 255.0));
    int q = (int)(v * (1 - f / 255.0));
    int t = (int)(v * (1 - (1 - f / 255.0) * s / 255.0));

    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return {r, g, b};
}

void save_normal(std::vector<std::array<float, 3> > normal, int W, int H, std::string path){
    cv::Mat image(H,W,CV_8UC3, cv::Scalar(0,0,0));

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int index = i * W + j;

            int r = (int)((normal[index][0] + 1) / 2 * 255);
            int b = (int)((normal[index][1] + 1) / 2 * 255);
            int g = (int)((normal[index][2] + 1) / 2 * 255);

        
            image.at<cv::Vec3b>(i,j) = cv::Vec3b(b,g,r);  // Blue, Green, Red
        }
    }

    cv::imwrite(path,image);
}

void save_mask(std::vector<int> mask, int W, int H, std::string path){
    cv::Mat image(H,W,CV_8UC3, cv::Scalar(0));

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int index = i * W + j;
            if (mask[index] != 0){
                std::array<int,3> color = hsv(mask[index], 4);
                image.at<cv::Vec3b>(i,j) = cv::Vec3b(color[0],color[1],color[2]);  // Blue, Green, Red
            } else {
                image.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);  // Black
            }
        }
    }

    cv::imwrite(path,image);
}

void save_cluster(std::vector<int> bin, int W, int H, std::string path){
    cv::Mat image(H,W,CV_8UC3, cv::Scalar(0));

    int max_bin = *std::max_element(bin.begin(), bin.end());

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int index = i * W + j;
            float ratio = (float)bin[index] / max_bin;
            image.at<cv::Vec3b>(i,j) = cv::Vec3b(
                (int)(ratio * 255), 
                (int)(ratio * 255), 
                (int)(ratio * 255)
            );  // Grayscale representation
        }
    }

    cv::imwrite(path,image);
}