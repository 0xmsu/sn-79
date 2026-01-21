add_library(xml INTERFACE)
add_library(taosim::xml ALIAS xml)

target_include_directories(xml INTERFACE include)

target_link_libraries(xml INTERFACE pugixml::static pugixml::pugixml)