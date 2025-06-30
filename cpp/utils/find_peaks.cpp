std::vector<int> find_peaks(
    std::vector<std::array<float,3> >& normal, 
    pcl::PointCloud<pcl::PointXYZRGB> points,
    std::array<float,3> grav,
    float bound,
    int kernel_size,
    int cluster_size,
    float plane_ratio = 0.02
){
    std::vector<float> dist(normal.size());

    float largest = -10000;
    float smallest = 10000;

    for(int i=0; i<normal.size(); ++i){
        if (dot(normal[i], grav)>bound){
            dist[i] = points[i].x * grav[0] + points[i].y * grav[1] + points[i].z * grav[2];
            if (largest<dist[i]) largest = dist[i];
            if (smallest>dist[i]) smallest = dist[i];
        }
    }

    //Cluster in cm bins
    float CLUSTER_SIZE = 0.01;

    std::vector<int> bins((int)((largest-smallest)/CLUSTER_SIZE)+1);

    for(int i=0; i<dist.size(); i++){
        if (dist[i]!=0){
            int index_i = (int)((dist[i]-smallest)/CLUSTER_SIZE);
            bins[index_i]++;
        }
    }

    std::vector<int> dillation(bins.size());
    for(int i=0; i<dillation.size(); i++){
        for(int j=std::max(i-kernel_size/2, 0); j<std::min(i+kernel_size/2+1,(int)dillation.size()); j++){
            dillation[i] = std::max(dillation[i], bins[j]);
        }
    }

    std::vector< std::pair<int,int> > store_index;
    for(int i=0; i<dillation.size(); i++){
        if (bins[i]==dillation[i] && bins[i]!=0){
            store_index.push_back(std::make_pair(i,bins[i]));
        }
    }

    std::partial_sort(
        store_index.begin(), store_index.begin(), store_index.end(),
        [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
            return a.second > b.second;  // Sort in descending order
        }
    );

    /*
    for(int i=0; i<store_index.size(); i++){
        std::cout << "Cluster " << i+1 << ": Index = " << store_index[i].first 
                  << ", Count = " << store_index[i].second 
                  << ", Distance = " << (store_index[i].first * CLUSTER_SIZE + smallest) 
                  << " m" << std::endl;
    }
    std::cout<<"\n";
    */

    std::vector<int> mask(normal.size());

    for(int i=0; i<(int)store_index.size(); i++){
        if (store_index[i].second < plane_ratio * normal.size()) {
            break; // Skip clusters that are too small
        }
        for(int j=0; j<normal.size(); j++){
            if (dist[j] != 0 && mask[j] == 0) { 
                int index_j = (int)((dist[j]-smallest)/CLUSTER_SIZE);
                if (index_j >= store_index[i].first - kernel_size/2 && index_j <= store_index[i].first + kernel_size/2) {
                    mask[j] = i + 1; // Assign cluster index (1-based)
                }
            }
        }
    }

    return mask;
}