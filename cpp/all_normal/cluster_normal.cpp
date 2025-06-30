void cluster_normal(std::vector< std::array<float, 3> > img_normals, int angle_bins = 36) {
    std::vector<int> bins(angle_bins * angle_bins, 0);
    for (int i = 0; i < img_normals.size(); ++i) {
        float angle_x = std::atan2(img_normals[i][1], img_normals[i][0]);
        float angle_y = std::atan2(img_normals[i][2], img_normals[i][0]);

        int bin_x = (int)((angle_x + M_PI) / (2 * M_PI) * angle_bins);
        int bin_y = (int)((angle_y + M_PI) / (2 * M_PI) * angle_bins);

        if (bin_x >= 0 && bin_x < angle_bins && bin_y >= 0 && bin_y < angle_bins) {
            bins[bin_x * angle_bins + bin_y]++;
        }
    }
    save_cluster(bins, angle_bins, angle_bins, "/catkin_ws/src/depth_plane_est/cluster_normal.png");
}
