#include <iostream>
#include <chrono>
#include <thread>


#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <Eigen/Geometry>
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

using namespace std;
using namespace pcl::console;
using namespace rs2;

typedef pcl::PointXYZRGB RGB_Cloud;
typedef pcl::PointCloud<RGB_Cloud> point_cloud;
typedef point_cloud::Ptr cloud_pointer;
pcl::visualization::PCLVisualizer::Ptr
simpleVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //viewer->addCoordinateSystem (1.0, "global");
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
    // cfg.enable_stream(RS2_STREAM_ACCEL);
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


rs2_vector gyro_data;
try{
  // auto frames = pipe.wait_for_frames();
  for (auto f : frames) {
     if (f.is<motion_frame>()) {
         motion_frame mf = f.as<motion_frame>();
         if (mf && mf.get_profile().stream_type() == RS2_STREAM_GYRO &&
    	mf.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
    {
        // Get gyro measurements
        gyro_data = mf.get_motion_data();
    }
         printf("%d:gyro_data: %f,%f,%f\n", f.get_profile().stream_type(),gyro_data.x ,gyro_data.y ,gyro_data.z);
         std::cout << "got motionframe" << '\n';
     }else{
       std::cout<<"not a poseframe"<<'\n';
     }
  }
}catch (const rs2::error & e)
  {
      std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;

  }



  pc.map_to(RGB);
  // Generate Point Cloud
  auto points = pc.calculate(depth);
  // Convert generated Point Cloud to PCL Formatting
  cloud_pointer cloud = PCL_Conversion(points, RGB);

  printf("PCL Pose: %f,%f,%f,%f\n",cloud->sensor_orientation_.x() ,cloud->sensor_orientation_.y() ,cloud->sensor_orientation_.z(),cloud->sensor_orientation_.w());
  cloud->sensor_orientation_=Eigen::Quaternionf()
  //========================================
  // Filter PointCloud (PassThrough Method)
  //========================================
  pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter; // Create the filtering object
  Cloud_Filter.setInputCloud (cloud);           // Input generated cloud to filter
  Cloud_Filter.setFilterFieldName ("z");        // Set field name to Z-coordinate
  Cloud_Filter.setFilterLimits (0.0, 1.0);      // Set accepted interval values
  Cloud_Filter.filter (*newCloud);              // Filtered Cloud Outputted

}

int main(int argc, char** argv)
{
  std::string arg1;
  // initialize PointClouds
  //cloud is input.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr final (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::RandomSampleConsensus<pcl::PointXYZRGB>* ransac;

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
  std::vector<int> inliers;

  // created RandomSampleConsensus object and compute the appropriated model
  pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>::Ptr
    model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZRGB> (cloud));
  pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloud));
  pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZRGB>::Ptr
    model_pp (new pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZRGB> (cloud));
  if(pcl::console::find_argument (argc, argv, "-m") >= 0)
  {
    switch (std::stoi(argv[pcl::console::find_argument (argc, argv, "-m")+1])) {
      case 1:
      ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_p);
      printf("RANSAC Model P\n" );
      break;
      case 2:
      ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_s);
      printf("RANSAC Model S\n" );
      break;
      case 3:
      model_pp->setAxis(Eigen::Vector3f (0.0, 0.0, 1.0));
      model_pp->setEpsAngle (pcl::deg2rad (15.0));
      ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_pp);
      printf("RANSAC Model Perpendicular Plane\n" );
      break;
      default:
      ransac =new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_p);
      printf("RANSAC Model P\n" );
    }
    if(pcl::console::find_argument (argc, argv, "-t") >= 0)
    {
      ransac->setDistanceThreshold (int(pcl::console::find_argument (argc, argv, "-t")+1));
    }else{
      ransac->setDistanceThreshold (0.01);
    }
    ransac->computeModel();
    ransac->getInliers(inliers);


  }else{

  }

  //     pcl::SampleConsensusModelSphere<pcl::PointXYZRGB>::Ptr
  //     model_p (new pcl::SampleConsensusModelSphere<pcl::PointXYZRGB> (cloud));
  //     pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_p);
  //     ransac.setDistanceThreshold (.01);
  // ransac.computeModel();
  // ransac.getInliers(inliers);
      // copies all inliers of the model computed to another PointCloud
      pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud, inliers, *final);
  // creates the visualization object and adds either our original cloud or all of the inliers
  // depending on the command line arguments specified.
  pcl::visualization::PCLVisualizer::Ptr viewer1;
  pcl::visualization::PCLVisualizer::Ptr viewer2;
  if (pcl::console::find_argument (argc, argv, "-m") >= 0){
    printf("Displaying processed cloud \n" );
    viewer1 = simpleVis(final);
    viewer2 = simpleVis(cloud);
  }else{
    printf("Displaying raw cloud\n");
    viewer1 = simpleVis(cloud);
  }
  while (!viewer1->wasStopped () && !viewer2->wasStopped ())
  {
    viewer1->spinOnce (100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return 0;
 }
