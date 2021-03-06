#' Extract frames from video for classification
#'
#' This function can take
#'
#' @param files dataframe of videos
#' @param outdir directory to save frames to
#' @param outfile file to which results will be saved
#' @param format output format for frames, defaults to jpg
#' @param fps frames per second, otherwise determine mathematically
#' @param frames number of frames to sample
#' @param parallel Toggle for parallel processing, defaults to FALSE
#' @param nproc number of processors to use if parallel, defaults to 1
#'
#' @return dataframe of still frames for each video
#' @export
#'
#' @examples
#' \dontrun{
#' frames <- imagesFromVideos(videos, outdir = "C:\\Users\\usr\\Videos\\", frames = 5)
#' }
imagesFromVideos <- function(files, outdir = tempfile(), outfile = NULL, format = "jpg", fps = NULL, frames = NULL, parallel = FALSE, nproc = 1) {
  if (checkFile(outfile)) { return(loadData(outfile))}
  if (!is(files, "data.frame")) { stop("Error: 'mdresults' must be Data Frame.")}
  if (outdir != "" & !dir.exists(outdir)) {
    if (!dir.create(outdir, recursive = TRUE)) { stop("Output directory invalid.\n")}
  }
  if (!is.null(fps) & !is.null(frames)) { message("If both fps and frames are defined fps will be used.")}
  if (is.null(fps) & is.null(frames)) { stop("Either fps or frames need to be defined.")}

  images <- files[tools::file_ext(files$FileName) %in% c("jpg", "JPG", "png", "PNG"), ]
  images$Frame <- images$FilePath
  videos <- files[tools::file_ext(files$FileName) %in% c("mp4", "MP4", "avi", "AVI"), ]

  run.parallel <- function(x, cond = "problem") {
    result <- tryCatch(
      {
        av::av_media_info(x)
      },
      error = function(e) {
        message(cond)
        message(e)
        return(NA)
      }
    )
    if (result$video$frames < 5) {
      files <- data.frame(FilePath = x, Frame = "File Corrupt")
    } else {
      vfilter <- ifelse(length(frames), paste0("fps=", round(1 / (result$duration / frames), 3)), "null")
      vfilter <- ifelse(length(fps), paste0("fps=", fps), vfilter)
      framerate <- result$video$framerate
      tempdir <- tempfile()
      dir.create(tempdir)
      name <- strsplit(basename(x), ".", fixed = T)[[1]][1]
      rnd <- sprintf("%05d", round(stats::runif(1, 1, 99999), 0))
      output <- file.path(tempdir, paste0(name, "_", rnd, "_%5d", ".jpg"))
      av::av_encode_video(input = x, output = output, framerate = framerate, vfilter = vfilter, verbose = F)
      files <- data.frame(FilePath = x, tmpframe = list.files(tempdir, pattern = paste0(name, "_", rnd, "_\\d{5}", ".jpg"), full.names = TRUE), stringsAsFactors = F)
      files$Frame <- paste0(outdir, basename(files$tmpframe))
      file.copy(files$tmpframe, files$Frame)
      file.remove(files$tmpframe)
      files[, c("FilePath", "Frame")]
    }
  }
  if (parallel) {
    type <- "PSOCK"

    cl <- parallel::makeCluster(min(parallel::detectCores(), nproc), type = type)
    parallel::clusterExport(cl, list("outdir", "format", "fps", "frames"), envir = environment())

    parallel::clusterSetRNGStream(cl)

    parallel::clusterEvalQ(cl, library(av))
    results <- pbapply::pblapply(videos$FilePath, function(x) {
      try(run.parallel(x))
    }, cl = cl)
    parallel::stopCluster(cl)
  } else {
    results <- pbapply::pblapply(videos$FilePath, function(x) {
      try(run.parallel(x))
    })
  }

  results <- do.call(rbind, results)
  videoframes <- merge(videos, results)
  allframes <- rbind(images, videoframes)

  # save frames to files
  if(!is.null(outfile)) { saveData(allframes, outfile)}

  allframes
}

