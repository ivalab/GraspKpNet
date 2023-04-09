cmake_minimum_required(VERSION 3.0.2)
project(gknet_perception)

find_package(catkin)

foreach(dir launch assets)
    install(
        DIRECTORY ${dir}/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir}
  )
endforeach(dir)

# add scripts
file(GLOB SCRIPT_FILES script/*)
install(
    PROGRAMS ${SCRIPT_FILES}
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/script
)

# add tests
if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    # for each test in the test directory add it as a test
    file(GLOB TEST_LAUNCH_FILES tests/*.test)
    foreach(TEST_LAUNCH_FILE ${TEST_LAUNCH_FILES})
        add_rostest(${TEST_LAUNCH_FILE})
    endforeach()
endif()
