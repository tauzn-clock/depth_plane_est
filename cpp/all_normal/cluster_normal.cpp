std::vector< std::pair<int, std::array<float, 3> > > cluster_normal(std::vector< std::array<float, 3> > img_normals, int angle_bins = 36, int angle_kernel_size = 5) {
    std::vector<int> bins(angle_bins * angle_bins, 0);
    for (int i = 0; i < img_normals.size(); ++i) {
        if (img_normals[i][0] == 0 && img_normals[i][1] == 0 && img_normals[i][2] == 0) continue;
        float angle_x = std::atan2(img_normals[i][0], img_normals[i][2]);
        float angle_y = std::atan2(img_normals[i][1], img_normals[i][2]);

        int bin_x = (int)((angle_x + M_PI/2) / (M_PI) * angle_bins);
        int bin_y = (int)((angle_y + M_PI/2) / (M_PI) * angle_bins);

        if (bin_x >= 0 && bin_x < angle_bins && bin_y >= 0 && bin_y < angle_bins) {
            bins[bin_x * angle_bins + bin_y]++;
        }
    }
    //save_cluster(bins, angle_bins, angle_bins, "/catkin_ws/src/depth_plane_est/cluster_normal.png");
    
    std::vector<int> kernel(angle_bins * angle_bins, 0);
    for(int i = 0; i < angle_bins; ++i) {
        for (int j = 0; j < angle_bins; ++j) {
            int index = i * angle_bins + j;
            kernel[index] = bins[index];  // Initialize kernel with the original bin values
            for (int ki = i - angle_kernel_size / 2; ki < i + angle_kernel_size / 2 + 1; ++ki) {
                for (int kj = j - angle_kernel_size / 2; kj < j + angle_kernel_size / 2 + 1; ++kj) {
                    kernel[index] = std::max(kernel[index], bins[(ki%angle_bins) * angle_bins + (kj%angle_bins)]);
                }
            }
        }
    }
    //save_cluster(kernel, angle_bins, angle_bins, "/catkin_ws/src/depth_plane_est/kernel.png");

    std::vector< std::pair<int, std::array<float, 3> > > normals;
    for (int i=0; i<angle_bins; i++) {
        for (int j=0; j<angle_bins; j++) {
            int index = i * angle_bins + j;
            if (bins[index] == kernel[index]) {
                float angle_x = (i + 0.5f) / angle_bins * M_PI - M_PI / 2;
                float angle_y = (j + 0.5f) / angle_bins * M_PI - M_PI / 2;

                std::array<float, 3> normal = {
                    std::tan(angle_x),
                    std::tan(angle_y),
                    1,
                };

                normalise(normal);

                normals.push_back(std::make_pair(bins[index], normal));
            }
        }
    }

    std::sort(normals.begin(), normals.end(),
        [](const std::pair<int, std::array<float, 3> >& a, const std::pair<int, std::array<float, 3> >& b) {
            return a.first > b.first;  // Sort in descending order
        });

    return normals;
}
