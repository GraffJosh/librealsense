#include <iostream>
#include <chrono>
#include <thread>
#include <future>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <Eigen/Geometry>
#include <mutex>
#include "../RS2_example.hpp"
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/visualization/pcl_visualizer.h>

#define PI 3.1415

using namespace std;
using namespace pcl::console;
using namespace rs2;

typedef pcl::PointXYZRGB RGB_Cloud;
typedef pcl::PointCloud<RGB_Cloud> point_cloud;
typedef point_cloud::Ptr cloud_pointer;

int algorithm;
float threshold;


class rotation_estimator
{
    // theta is the angle of camera rotation in x, y and z components
    float3 theta;
    std::mutex theta_mtx;
    /* alpha indicates the part that gyro and accelerometer take in computation of theta; higher alpha gives more weight to gyro, but too high
    values cause drift; lower alpha gives more weight to accelerometer, which is more sensitive to disturbances */
    float alpha = 0.98;
    bool first = true;
    // Keeps the arrival time of previous gyro frame
    double last_ts_gyro = 0;
public:
    // Function to calculate the change in angle of motion based on data from gyro
    void process_gyro(rs2_vector gyro_data, double ts)
    {
        if (first) // On the first iteration, use only data from accelerometer to set the camera's initial position
        {
            last_ts_gyro = ts;
            return;
        }
        // Holds the change in angle, as calculated from gyro
        float3 gyro_angle;

        // Initialize gyro_angle with data from gyro
        gyro_angle.x = gyro_data.x; // Pitch
        gyro_angle.y = gyro_data.y; // Yaw
        gyro_angle.z = gyro_data.z; // Roll

        // Compute the difference between arrival times of previous and current gyro frames
        double dt_gyro = (ts - last_ts_gyro) / 1000.0;
        last_ts_gyro = ts;

        // Change in angle equals gyro measures * time passed since last measurement
        gyro_angle = gyro_angle * dt_gyro;

        // Apply the calculated change of angle to the current angle (theta)
        std::lock_guard<std::mutex> lock(theta_mtx);
        theta.add(-gyro_angle.z, -gyro_angle.y, gyro_angle.x);
    }

    void process_accel(rs2_vector accel_data)
    {
        // Holds the angle as calculated from accelerometer data
        float3 accel_angle;

        // Calculate rotation angle from accelerometer data
        accel_angle.z = atan2(accel_data.y, accel_data.z);
        accel_angle.x = atan2(accel_data.x, sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z));

        // If it is the first iteration, set initial pose of camera according to accelerometer data (note the different handling for Y axis)
        std::lock_guard<std::mutex> lock(theta_mtx);
        if (first)
        {
            first = false;
            theta = accel_angle;
            // Since we can't infer the angle around Y axis using accelerometer data, we'll use PI as a convetion for the initial pose
            theta.y = PI;
        }
        else
        {
            /*
            Apply Complementary Filter:
                - high-pass filter = theta * alpha:  allows short-duration signals to pass through while filtering out signals
                  that are steady over time, is used to cancel out drift.
                - low-pass filter = accel * (1- alpha): lets through long term changes, filtering out short term fluctuations
            */
            theta.x = theta.x * alpha + accel_angle.x * (1 - alpha);
            theta.z = theta.z * alpha + accel_angle.z * (1 - alpha);
        }
    }

    // Returns the current rotation angle
    float3 get_theta()
    {
        std::lock_guard<std::mutex> lock(theta_mtx);
        return theta;
    }
};

pcl::visualization::PCLVisualizer::Ptr
simpleVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,string name)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (name));
  viewer->setBackgroundColor (0, 0, 0);
  // viewer->addCoordinateSystem (1.0);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, name);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, name);
  // viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  return (viewer);
}


//======================================================
// RGB Texture
// - Function is utilized to extract the RGB data from
// a single point return R, G, and B values.
// Normals are stored as RGB components and
// correspond to the specific depth (XYZ) coordinate.
// By taking these normals and converting them to
// texture coordinates, the RGB components can be
// "mapped" to each individual point (XYZ).
//======================================================
std::tuple<int, int, int> RGB_Texture(rs2::video_frame texture, rs2::texture_coordinate Texture_XY)
{
    // Get Width and Height coordinates of texture
    int width  = texture.get_width();  // Frame width in pixels
    int height = texture.get_height(); // Frame height in pixels

    // Normals to Texture Coordinates conversion
    int x_value = min(max(int(Texture_XY.u * width  + .5f), 0), width - 1);
    int y_value = min(max(int(Texture_XY.v * height + .5f), 0), height - 1);

    int bytes = x_value * texture.get_bytes_per_pixel();   // Get # of bytes per pixel
    int strides = y_value * texture.get_stride_in_bytes(); // Get line width in bytes
    int Text_Index =  (bytes + strides);

    const auto New_Texture = reinterpret_cast<const uint8_t*>(texture.get_data());

    // RGB components to save in tuple
    int NT1 = New_Texture[Text_Index];
    int NT2 = New_Texture[Text_Index + 1];
    int NT3 = New_Texture[Text_Index + 2];

    return std::tuple<int, int, int>(NT1, NT2, NT3);
}
//===================================================
//  PCL_Conversion
// - Function is utilized to fill a point cloud
//  object with depth and RGB data from a single
//  frame captured using the Realsense.
//===================================================
cloud_pointer PCL_Conversion(const rs2::points& points, const rs2::video_frame& color){

    // Object Declaration (Point Cloud)
    cloud_pointer cloud(new point_cloud);

    // Declare Tuple for RGB value Storage (<t0>, <t1>, <t2>)
    std::tuple<uint8_t, uint8_t, uint8_t> RGB_Color;

    //================================
    // PCL Cloud Object Configuration
    //================================
    // Convert data captured from Realsense camera to Point Cloud
    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width  = static_cast<uint32_t>( sp.width()  );
    cloud->height = static_cast<uint32_t>( sp.height() );
    cloud->is_dense = false;
    cloud->points.resize( points.size() );

    auto Texture_Coord = points.get_texture_coordinates();
    auto Vertex = points.get_vertices();

    // Iterating through all points and setting XYZ coordinates
    // and RGB values
    for (int i = 0; i < points.size(); i++)
    {
        //===================================
        // Mapping Depth Coordinates
        // - Depth data stored as XYZ values
        //===================================
        cloud->points[i].x = Vertex[i].x;
        cloud->points[i].y = Vertex[i].y;
        cloud->points[i].z = Vertex[i].z;

        // Obtain color texture for specific point
        RGB_Color = RGB_Texture(color, Texture_Coord[i]);

        // Mapping Color (BGR due to Camera Model)
        cloud->points[i].r = get<2>(RGB_Color); // Reference tuple<2>
        cloud->points[i].g = get<1>(RGB_Color); // Reference tuple<1>
        cloud->points[i].b = get<0>(RGB_Color); // Reference tuple<0>

    }

   return cloud; // PCL RGB Point Cloud generated
}

Eigen::Quaternionf toQuaternion( double yaw, double pitch, double roll) // yaw (Z), pitch (Y), roll (X)
{
    // Abbreviations for the various angular functions
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Eigen::Quaternionf q;
    q.w() = cy * cp * cr + sy * sp * sr;
    q.x() = cy * cp * sr - sy * sp * cr;
    q.y() = sy * cp * sr + cy * sp * cr;
    q.z() = sy * cp * cr - cy * sp * sr;
    return q;
}
float3 toEulerAngle(Eigen::Quaternionf q)
{
  float3 euler;
	// roll (x-axis rotation)
	double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
	double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
	euler.x = atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
	if (fabs(sinp) >= 1)
		euler.y = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
	else
		euler.y = asin(sinp);

	// yaw (z-axis rotation)
	double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
	double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
	euler.z = atan2(siny_cosp, cosy_cosp);
  return euler;
}

void getCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr newCloud)
{
  // Declare pointcloud object, for calculating pointclouds and texture mappings
  rs2::pointcloud pc;

  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;

  // Create a configuration for configuring the pipeline with a non default profile
  rs2::config cfg;

  cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
  cfg.enable_stream(RS2_STREAM_INFRARED, 1280, 720, RS2_FORMAT_Y8, 30);
  cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
  cfg.enable_stream(RS2_STREAM_ACCEL);
  cfg.enable_stream(RS2_STREAM_GYRO);
  // cfg.enable_stream(RS2_STREAM_POSE, RS2_FORMAT_6DOF);
  rs2::pipeline_profile selection = pipe.start(cfg);

  rs2::device selected_device = selection.get_device();
  auto depth_sensor = selected_device.first<rs2::depth_sensor>();

  if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED))
  {
      depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
      //depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
  }
  if (depth_sensor.supports(RS2_OPTION_LASER_POWER))
  {
      // Query min and max values:
      auto range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
      depth_sensor.set_option(RS2_OPTION_LASER_POWER, range.max); // Set max power
      //depth_sensor.set_option(RS2_OPTION_LASER_POWER, 0.f); // Disable laser
  }

  std::cout << "Enabled Stream" << '\n';
  // Capture a single frame and obtain depth + RGB values from it
  // Wait for frames from the camera to settle
 for (int i = 0; i < 10; i++) {
     auto frames = pipe.wait_for_frames(); //Drop several frames for auto-exposure
 }


 std::cout << "waited for 30 frames" << '\n';
 auto frames = pipe.wait_for_frames();
  auto depth = frames.get_depth_frame();
  auto RGB = frames.get_color_frame();
  std::cout << "rgb Frame collected." << '\n';


  // Declare object that handles camera pose calculations
  rotation_estimator algo;
try{
  // auto frames = pipe.wait_for_frames();
  for (auto f : frames) {
       if (f.is<motion_frame>()) {
          // Cast the frame that arrived to motion frame
          auto motion = f.as<rs2::motion_frame>();
          // If casting succeeded and the arrived frame is from gyro stream
          if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
          {
              // Get the timestamp of the current frame
              double ts = motion.get_timestamp();
              // Get gyro measures
              rs2_vector gyro_data = motion.get_motion_data();
              // Call function that computes the angle of motion based on the retrieved measures
              //algo.process_gyro(gyro_data, ts);
          }
          // If casting succeeded and the arrived frame is from accelerometer stream
          if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
          {
              // Get accelerometer measures
              rs2_vector accel_data = motion.get_motion_data();
              // Call function that computes the angle of motion based on the retrieved measures
              algo.process_accel(accel_data);
          }
       }else{
         // std::cout<<"not a poseframe"<<'\n';
       }
    }
  }catch (const rs2::error & e)
  {
      std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;

  }
  printf("accel_data: x:%f,y:%f,z:%f\n",algo.get_theta().x*(180/PI),algo.get_theta().y*(180/PI) ,algo.get_theta().z*(180/PI));
  std::cout << "got motionframe" << '\n';


  pc.map_to(RGB);
  // Generate Point Cloud
  auto points = pc.calculate(depth);
  // Convert generated Point Cloud to PCL Formatting
  cloud_pointer cloud = PCL_Conversion(points, RGB);
  cloud->sensor_origin_ = {0,0,0,0};
  cloud->sensor_orientation_ = toQuaternion(0-algo.get_theta().x-PI,PI-algo.get_theta().y,algo.get_theta().z+(PI/2));
  printf("PCL Pose: %f,%f,%f\n",toEulerAngle(cloud->sensor_orientation_).x*(180/PI) ,toEulerAngle(cloud->sensor_orientation_).y*(180/PI) ,toEulerAngle(cloud->sensor_orientation_).z*(180/PI));
  printf("PCL Pose: %f,%f,%f,%f\n",cloud->sensor_orientation_.x() ,cloud->sensor_orientation_.y() ,cloud->sensor_orientation_.z(),cloud->sensor_orientation_.w());
  //========================================
  // Filter PointCloud (PassThrough Method)
  //========================================
  pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter; // Create the filtering object
  Cloud_Filter.setInputCloud (cloud);           // Input generated cloud to filter
  Cloud_Filter.setFilterFieldName ("z");        // Set field name to Z-coordinate
  Cloud_Filter.setFilterLimits (0.0, 1.0);      // Set accepted interval values
  Cloud_Filter.filter (*newCloud);              // Filtered Cloud Outputted

}

void process_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr final,int algorithm, float threshold)
{
  std::vector<int> inliers;

  pcl::RandomSampleConsensus<pcl::PointXYZRGB>* ransac;
  // created RandomSampleConsensus object and compute the appropriated model
  pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>::Ptr
    model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZRGB> (cloud));
  pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloud));
  pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZRGB>::Ptr
    model_pp (new pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZRGB> (cloud));

  if(algorithm > 0)
  {
    switch (algorithm) {
      case 1:
      printf("RANSAC Model P\n" );
      ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_p);
      break;
      case 2:
      printf("RANSAC Model S\n" );
      ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_s);
      break;
      case 3:
      // IMU_Orientation = cloud->sensor_orientation_.toRotationMatrix().eulerAngles(0,1,2);
      printf("RANSAC Model Perpendicular Plane\n" );
      model_pp->setAxis(Eigen::Vector3f (toEulerAngle(cloud->sensor_orientation_).y*(180/PI),0-toEulerAngle(cloud->sensor_orientation_).x*(180/PI)-90 ,toEulerAngle(cloud->sensor_orientation_).z*(180/PI)));
      std::cout << "axis: "<<model_pp->getAxis() << '\n';
      model_pp->setEpsAngle (pcl::deg2rad (5.0));
      ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_pp);
      break;
      default:
      printf("RANSAC Model P\n" );
      ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_p);
    }
  }else{
    printf("RANSAC Model P\n" );
    ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_p);
  }

  ransac->setDistanceThreshold (threshold);
  ransac->computeModel();
  ransac->getInliers(inliers);
  pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud, inliers, *final);
  printf("Number of inliers:%d\n", inliers.size());
  printf("Processed size:%d -> %d\n", cloud->size(),final->size());
}

int main(int argc, char** argv)
{
  std::string arg1;
  // initialize PointClouds
  //cloud is input.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr final (new pcl::PointCloud<pcl::PointXYZRGB>);

  // pcl::visualization::PCLVisualizer::Ptr viewer1;
  // pcl::visualization::PCLVisualizer::Ptr viewer2;
  pcl::visualization::PCLVisualizer::Ptr viewer1 (new pcl::visualization::PCLVisualizer ("base"));
  pcl::visualization::PCLVisualizer::Ptr viewer2 (new pcl::visualization::PCLVisualizer ("final"));
  viewer1->setBackgroundColor (0, 0, 0);
  viewer1->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "base");
  viewer1->initCameraParameters ();
  viewer2->setBackgroundColor (0, 0, 0);
  viewer2->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "final");
  viewer2->initCameraParameters ();

  if(pcl::console::find_argument (argc, argv, "-f") >= 0)
  {
    auto openFileName = argv[find_argument(argc, argv, "-f")+1];
    printf("filename: %s\n", openFileName );
    //auto openFileName = "Captured_Frame" + to_string(1) + ".pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (openFileName, *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file %s\n",openFileName);
    return (-1);
  }
  }else{
    printf("GetCloud\n" );
    getCloud(cloud);
  }

  if(pcl::console::find_argument (argc, argv, "-t") >= 0)
  {
    threshold = (std::stof(argv[pcl::console::find_argument (argc, argv, "-t")+1]));
  }else{
    threshold = .01;
  }

  if (pcl::console::find_argument (argc, argv, "-m") > 0) {
    /* code */
    algorithm = std::stoi(argv[pcl::console::find_argument (argc, argv, "-m")+1]);
    printf("process cloud, algorithm: %d\n", algorithm);
  }else{
    algorithm = 0;
  }


  process_cloud(cloud,final,algorithm,threshold);


  while (!viewer1->wasStopped () && !viewer2->wasStopped ())
  {
    // Execute lambda asyncronously.
    auto future = std::async(std::launch::async, [] {
        std::cout << "press enter to collect another image" << '\n';
        getchar();
        return;
    });
    printf("displayed size: size:%d -> %d\n", cloud->size(),final->size());
    viewer1->removePointCloud("base");
    viewer1->addPointCloud<pcl::PointXYZRGB>(cloud,"base");
    viewer2->removePointCloud("final");
    viewer2->addPointCloud<pcl::PointXYZRGB>(final,"final");
    viewer2->addCoordinateSystem(.5);
    // Continue execution in main thread.
    auto status = future.wait_for(std::chrono::milliseconds(0));
    while(status != std::future_status::ready) {
        status = future.wait_for(std::chrono::milliseconds(0));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        viewer1->spinOnce (100);
        viewer2->spinOnce (100);
    }
    getCloud(cloud);
    process_cloud(cloud,final,algorithm,threshold);

  }
  return 0;
 }
