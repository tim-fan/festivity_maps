# ToDo

* improve gps processing, and path distance
  * Ideally some sort of fusion between the two cameras to get consistent distance and heading over time
    * Use gtsam?
       * nav state?
* in fiftyone, sort by image timestamp (from exif)
* in fiftyone, view heatmap as color by value
* CLI laggy to start up - probably importing everything
* when scoring single image by path, dont save heatmap or update db?
* script to view heatmap + source image side by side
* duplication between score / train
  * resize_transform
  * running inference
  * model loading, MODEL_TO_NUM_LAYERS
* full workflow test with test data
* score single image (full path provided) fails when image not in workspace
* add live location to map?
* rename categories to:
  * no lights detected
  * some lights detected
  * moderate lights detected
  * many lights detected
* add map screenshot to readme
* workspace setup/install instructions
* trajectory stats - how many properties, how many were decorated at different level
* fix thresholds to compare counts across trajectories
* analysis - best street, best 100m, 
* change size of images
* reconsider the need to do exif rotations on import
* remove address clustering from map command, move to gps command
* can batch process while scoring?
* continuous colorscale + size
* fest map jumps between trajectories
* glow effect makes neighbours unclickable
* 360 camera with depth processesing https://github.com/Insta360-Research-Team/DAP
