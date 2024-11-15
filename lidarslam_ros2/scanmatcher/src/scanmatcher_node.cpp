#include "scanmatcher/scanmatcher_component.h"
#include <rclcpp/rclcpp.hpp>
#include <pcl/console/print.h>

int main(int argc, char * argv[])
{
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  rclcpp::spin(std::make_shared<graphslam::ScanMatcherComponent>(options));
  rclcpp::shutdown();
  return 0;
}
