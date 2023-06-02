# Seek 'n' Functions
Programs that traverse through a source directory and its subdirectories, searching for files and executing a specific action on each of them (such as conversions, plotting, etc.). When dealing with experiments and applications that produce multiple similar files, extracting results and plotting results can be a boring and tiring experience. The functions stored here were written to automate these tasks. They arose from the need speed up my day to day research work, hence it might be helpful in similar tasks. The main functions are detailed below.

### seek_n_comp functions
These functions search for files that meet a specific criteria, such as having the same suffix/prefix or matching a regex pattern. They then perform a desired action, defined by an user's function. There are two implementations:
  (1) seek_n_comp_1 that works with a single file at a time;
  (2) seek_n_comp_2 that works with two files, assuming that one of them is a reference file, whereas the other contains the data that we want to compare with the former.

### seek_n_convert function
Thiis function is similar to the one above, but is tailored to file conversion.

### aux_functions
This module contains the implementations of functions that are compatible with the "seek_n_comp" functions. It is important to ensure that all functions adhere to the specified variable structure present in these functions. If one desires to employ a function that doesn't follow this structure, modifications must be made to the "seeker" functions at the locations where the functions are called.
