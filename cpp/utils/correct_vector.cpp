void correct_vector(
    std::vector<std::array<float,3> > normal, 
    std::array<float,3>& grav,
    float bound,
    int iteration
){
    for(int iteration_ = 0; iteration_<iteration; iteration_++){
        int cnt=0;
        float mean_x=0;
        float mean_y=0;

        std::array<float,3> grav_x = {0,grav[1], grav[2]};
        std::array<float,3> grav_y = {grav[0],0, grav[2]};
        normalise(grav_x);
        normalise(grav_y);
        
        std::array<float,3> normal_x, normal_y;

        for (int i = 0; i < normal.size(); ++i) {
            if (dot(normal[i], grav) > bound && normal[i] != std::array<float,3>{0,0,0}) {
                normal_x = normal[i];
                normal_x[0] = 0;
                normalise(normal_x);

                normal_y = normal[i];
                normal_y[1] = 0;
                normalise(normal_y);

                float cos_x = normal_x[1] * grav_x[1] + normal_x[2] * grav_x[2];
                float sin_x = normal_x[2] * grav_x[1] - normal_x[1] * grav_x[2];
                float cos_y = normal_y[0] * grav_y[0] + normal_y[2] * grav_y[2];
                float sin_y = - normal_y[2] * grav_y[0] + normal_y[0] * grav_y[2];

                float angle_x = std::atan2(sin_x, cos_x);
                float angle_y = std::atan2(sin_y, cos_y);

                mean_x += angle_x;
                mean_y += angle_y;
                cnt++;
            }
        }

        if (cnt == 0) break;
        mean_x /= cnt;
        mean_y /= cnt;

        rotation(grav, mean_x, mean_y, 0);
    }
}