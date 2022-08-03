# -*- coding: utf-8 -*-
"""
Created on Jun 23 2022

@author: hejar Shahabi
Email: hejarshahabi@gmail.com
"""
import numpy as np
import rasterio as rs
import glob
import os 
import tqdm
import matplotlib.pyplot as plt


class Generator:
    """
    Using the package, you will be able to turn 
    satellite imagery into patches with various shape sizes
    and apply data augmentation techniques such as flip and rotation.
    
    parameters:
    -----------
    image:              NumPy array, or string that
                        shows path to satellite imagery with "tif" format
    
    label:              NumPy array, or string that
                        shows path to label raster with "tif" format
                
    patch_size:         Integer
                        moving window size to generate image patches
                        From example: for a 128*128 patch just 128 needs to be passed
                
    stride:             Integer
                        For small images or large patch size it is recommended to have
                        a stride value less than patch size to generate more image patches
                        and also having overly with consecutive patches. in case of no overly pass
                        stride value equal or bigger than patch size.
                    
    channel_first:     Boolean 
                       True if the first index is the number of image bands and False if it is the last index.
    
    """

    def __init__(self, image=None, label=None, patch_size=None, stride=None, channel_first=True):
        
        self.image=image
        self.label=label
        self.patch_size=patch_size
        self.stride=stride
        self.channel=channel_first
    def readData(self):
        if type(self.image)==str and self.image[-3:]=="tif":
            self.img=rs.open(self.image)
            self.band=self.img.count 
            self.row =self.img.width
            self.col = self.img.height 
            self.imgarr=self.img.read()
            self.shape=self.imgarr.shape
            self.inv=rs.open(self.label)
            self.inv_row =self.inv.width
            self.inv_col = self.inv.height 
            self.inv=self.inv.read(1)
            if self.channel==True:
                self.imgarr=self.imgarr
            else:
                self.imgarr=np.transpose(self.imgarr,(2,0,1))
            return self.imgarr, self.inv
           
        elif type(self.image)==str and self.image[-3:]=="npy":
            self.imgarr=np.load(self.image)
            self.shape=self.imgarr.shape
            self.inv=np.load(self.label)
            if self.channel==True:
                self.imgarr=self.imgarr
            else:
                self.imgarr=np.transpose(self.imgarr,(2,0,1))
            return self.imgarr, self.inv
            
        else:
            self.shape=self.image.shape
            if self.channel==True:
                self.imgarr=self.image
            else:
                self.imgarr=np.transpose(self.image,(2,0,1))
                
            self.inv=self.label
            return self.imgarr, self.inv, self.shape
       
    
    def data_dimension(self):
        """
        To display input data dimension
        """
        self.readData()
        print("############################################")
        print(f' the shape of image is: {self.shape} ')
        print(f' the shape of label is: {self.inv.shape} ')
        print("############################################")

    def patch_info(self):
        """
        To isplay the number of orginal image patches can be 
        generated based on given patch size and stride values.
        """
        self.readData()
        x_dim=self.imgarr.shape[1]
        y_dim=self.imgarr.shape[2]
     
        self.X=((x_dim-self.patch_size)//self.stride)*self.stride
        self.Y=((y_dim-self.patch_size)//self.stride)*self.stride
        self.covx=self.X+self.patch_size
        self.covy=self.Y+self.patch_size
        self.total_patches= ((x_dim-self.patch_size)//self.stride)*((y_dim-self.patch_size)//self.stride)
       
    
        print("       ")
        print("#######################################################################################")
        print(f'The effective X-dimension of the input data for patch generation is from 0 to {self.covx}')
        print(f'The effective Y-dimension of the input data for patch generation is from 0 to {self.covy}')
        print(f'The number of total non-augmented patches that can be generated\n'
              f'based on patch size ({self.patch_size}*{self.patch_size}) and stride ({self.stride}) is "{self.total_patches}"')
        
        print("#######################################################################################")
        print("                                                                        ")
        #print("#######################################################################################")

    
    def save_Geotif(self, folder_name="tif", only_label=False):
        #self.patch_info()
        

        """
        
        parameters:
        -----------
        folder_name:        String
                            Passing the folder name as string like "tif" so a folder 
                            with that name will be generated in the current working directory 
                            to save Geotif image patches there in the sub-folders called "patch" and "label"
        
        only_label:         Boolean
                            If True, only the patches will be exported that have labelled data.

        """
        
        if not os.path.exists(folder_name+"/patch"):
                                  os.makedirs(folder_name+"/patch")
                                  os.makedirs(folder_name+"/label")
                          
        with tqdm.tqdm(range(self.total_patches), desc="Patch Counter", unit=" Patch") as pbar:
            index=1
            patch_counter=0
            for i in (range(0,self.X,self.stride)):
                for j in (range(0,self.Y,self.stride)):
                    self.img_patch= self.imgarr[:,i:i+self.patch_size,j:j+self.patch_size]
                    self.lbl_patch= self.inv[i:i+self.patch_size,j:j+self.patch_size]
                    if only_label==True:
                        if self.lbl_patch.any()<1:
                            continue
                        self.x_cord,self.y_cord = (j*self.img.transform[0]+self.img.transform[2]),(self.img.transform[5]+i*self.img.transform[4])
                        transform= [self.img.transform[0],0,self.x_cord,0,self.img.transform[4],self.y_cord]
                            
                        with rs.open(folder_name+"/patch/"+str(index)+"_img.tif","w",driver='GTiff', count=self.band, dtype=self.imgarr.dtype,
                                          width=self.patch_size, height=self.patch_size, transform=transform, crs=self.img.crs) as raschip:
                                raschip.write(self.img_patch)
                        with rs.open(folder_name+"/label/"+str(index)+"_lbl.tif","w",driver='GTiff', count=1, dtype=self.inv.dtype,
                                          width=self.patch_size, height=self.patch_size, transform=transform, crs=self.img.crs) as lblchip:
                                lblchip.write(self.lbl_patch,1)
                                patch_counter+=1
                        index+=1
                        pbar.update(1)
                            
                    if only_label==False:
                        
                        self.x_cord,self.y_cord = (j*self.img.transform[0]+self.img.transform[2]),(self.img.transform[5]+i*self.img.transform[4])
                        transform= [self.img.transform[0],0,self.x_cord,0,self.img.transform[4],self.y_cord]
                            
                        with rs.open(folder_name+"/patch/"+str(index)+"_img.tif","w",driver='GTiff', count=self.band, dtype=self.imgarr.dtype,
                                          width=self.patch_size, height=self.patch_size, transform=transform, crs=self.img.crs) as raschip:
                                raschip.write(self.img_patch)
                        with rs.open(folder_name+"/label/"+str(index)+"_lbl.tif","w",driver='GTiff', count=1, dtype=self.inv.dtype,
                                          width=self.patch_size, height=self.patch_size, transform=transform, crs=self.img.crs) as lblchip:
                                lblchip.write(self.lbl_patch,1)
                                patch_counter+=1
                
                        index+=1
                        pbar.update(1)
            total_patches=pbar.total
            save_patches=patch_counter
            percentage= int((save_patches/total_patches)*100)
            print("#######################################################################################")
            print(f'{save_patches} patches, which are %{percentage} of totall patches, are saved as ".tif" format in "{os.getcwd()}\{folder_name}"')
            print("#######################################################################################")


    def save_numpy(self, folder_name="npy", only_label=True, return_stacked=True, save_stack=True,  V_flip=True, H_flip=True, Rotation=True):
        #self.patch_info()
        #self.patch_info()
        """
        Using this function image patches will be generated in NumPy format with data augmentation options.
        
        parameters :
        -----------
        folder_name:        String
                            Passing the folder name as string like "npy" so a folder 
                            with that name will be generated in the current working directory 
                            to save image patches in NumPy format there in the sub-folders called "patch" and "label"
        
        only_label:         Boolean
                            If True, only the patches will be exported that have labelled data.
                    
        Return_stacked:     Boolean
                            if True, the generated patches will be stacked to gather as a NumPy array
                     
        save_stacked :     Boolean 
                            If True, the stacked arrays of patches and labels 
                            will be save as NumPy array in "folder_name" directory
                            
        V_flip:             Boolean
                            if True, Vertical flip will be applied on image patches/labels for data augmentation
                                  
        H_flip:             Boolean 
                            if True, Horizontal flip will be applied on image patches/labels for data augmentation
                            
        Rotation:           Boolean 
                            if True, 90-, 180- and 270-degree rotations will be applied on image patches/labels for data augmentation
                        
                        
        
        """

        
        if not os.path.exists(folder_name+"/patch"):
                                  os.makedirs(folder_name+"/patch")
                                  os.makedirs(folder_name+"/label")
        num=[0]
        if V_flip==True:
            num.append("V")
        if H_flip==True:
            num.append("H")
        if Rotation==True:
            num.append("90")
            num.append("180")
            num.append("270")
        x=[]
        y=[]
        if only_label==True:
            with tqdm.tqdm(range(self.total_patches*len(num)), desc="Patch Counter", unit=" Patch") as pbar:
                index=1
                patch_counter=0
    
                for i in (range(0,self.X,self.stride)):
                    for j in (range(0,self.Y,self.stride)):
                        self.img_patch= self.imgarr[:,i:i+self.patch_size,j:j+self.patch_size]
                        self.lbl_patch= self.inv[i:i+self.patch_size,j:j+self.patch_size]
                        if self.lbl_patch.any()<1:
                            continue
                        self.img_patch=np.transpose(self.img_patch, (1,2,0))
                        np.save(folder_name+"/patch/"+str(index)+"_img.npy", self.img_patch)
                        np.save(folder_name+"/label/"+str(index)+"_lbl.npy", self.lbl_patch)
                        patch_counter+=1
                        pbar.update(1)
                        if return_stacked==True:
                            x.append(self.img_patch)
                            y.append(self.lbl_patch)
                        
                        if V_flip==True:
                            np.save(folder_name+"/patch/"+str(index)+"_img_V.npy", np.flip(self.img_patch, axis=0))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_V.npy", np.flip(self.lbl_patch, axis=0))
                            patch_counter+=1
                            pbar.update(1)
                            if return_stacked==True:
                                x.append(np.flip(self.img_patch, axis=0))
                                y.append(np.flip(self.lbl_patch, axis=0))
                            
                        if H_flip==True:
                            np.save(folder_name+"/patch/"+str(index)+"_img_H.npy", np.flip(self.img_patch, axis=1))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_H.npy",  np.flip(self.lbl_patch, axis=1))
                            patch_counter+=1
                            pbar.update(1)
                            if return_stacked==True:
                                x.append(np.flip(self.img_patch, axis=1))
                                y.append(np.flip(self.lbl_patch, axis=1))
                        if Rotation==True:
                            np.save(folder_name+"/patch/"+str(index)+"_img_90.npy", np.rot90(self.img_patch,  k=1, axes=(0, 1)))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_90.npy", np.rot90(self.lbl_patch,  k=1, axes=(0, 1)))
                            np.save(folder_name+"/patch/"+str(index)+"_img_180.npy", np.rot90(self.img_patch, k=2, axes=(0, 1)))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_180.npy", np.rot90(self.lbl_patch, k=2, axes=(0, 1)))
                            np.save(folder_name+"/patch/"+str(index)+"_img_270.npy", np.rot90(self.img_patch, k=3, axes=(0, 1)))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_270.npy", np.rot90(self.lbl_patch, k=3, axes=(0, 1)))
                            if return_stacked==True:
                                x.append(np.rot90(self.img_patch,  k=1, axes=(0, 1)))
                                y.append(np.rot90(self.lbl_patch,  k=1, axes=(0, 1)))
                                x.append(np.rot90(self.img_patch,  k=2, axes=(0, 1)))
                                y.append(np.rot90(self.lbl_patch,  k=2, axes=(0, 1)))
                                x.append(np.rot90(self.img_patch,  k=3, axes=(0, 1)))
                                y.append(np.rot90(self.lbl_patch,  k=3, axes=(0, 1)))

                            
                            
                            patch_counter+=3
                            pbar.update(3) 
                        index+=1
            total_patches=pbar.total
            save_patches=patch_counter
            percentage= int((save_patches/total_patches)*100)
    
            print("#######################################################################################")
            print(f"{save_patches} patches, which are %{percentage} of total patches,\n"
                  f'have the labelled data and are saved as ".npy" format in "{os.getcwd()}\{folder_name}"')
    
            print("#######################################################################################")
        
        if only_label==False:
            with tqdm.tqdm(range(self.total_patches*len(num)), desc="Patch Counter", unit=" Patch") as pbar:
                index=1
                for i in (range(0,self.X,self.stride)):
                    for j in (range(0,self.Y,self.stride)):
                        self.img_patch= self.imgarr[:,i:i+self.patch_size,j:j+self.patch_size]
                        self.lbl_patch= self.inv[i:i+self.patch_size,j:j+self.patch_size]
                        self.img_patch=np.transpose(self.img_patch, (1,2,0))
                        np.save(folder_name+"/patch/"+str(index)+"_img.npy", self.img_patch)
                        np.save(folder_name+"/label/"+str(index)+"_lbl.npy", self.lbl_patch)
                        pbar.update(1)
                        if return_stacked==True:
                            x.append(self.img_patch)
                            y.append(self.lbl_patch)
                        
                        
                        if V_flip==True:
                            np.save(folder_name+"/patch/"+str(index)+"_img_V.npy", np.flip(self.img_patch, axis=0))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_V.npy", np.flip(self.lbl_patch, axis=0))
                            pbar.update(1)
                            if return_stacked==True:
                                x.append(np.flip(self.img_patch, axis=0))
                                y.append(np.flip(self.lbl_patch, axis=0))
                        if H_flip==True:
                            np.save(folder_name+"/patch/"+str(index)+"_img_H.npy", np.flip(self.img_patch, axis=1))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_H.npy",  np.flip(self.lbl_patch, axis=1))
                            if return_stacked==True:
                                x.append(np.flip(self.img_patch, axis=1))
                                y.append(np.flip(self.lbl_patch, axis=1))
                            pbar.update(1)
                        if Rotation==True:
                            np.save(folder_name+"/patch/"+str(index)+"_img_90.npy", np.rot90(self.img_patch,  k=1, axes=(0, 1)))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_90.npy", np.rot90(self.lbl_patch,  k=1, axes=(0, 1)))
                            np.save(folder_name+"/patch/"+str(index)+"_img_180.npy", np.rot90(self.img_patch, k=2, axes=(0, 1)))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_180.npy", np.rot90(self.lbl_patch, k=2, axes=(0, 1)))
                            np.save(folder_name+"/patch/"+str(index)+"_img_270.npy", np.rot90(self.img_patch, k=3, axes=(0, 1)))
                            np.save(folder_name+"/label/"+str(index)+"_lbl_270.npy", np.rot90(self.lbl_patch, k=3, axes=(0, 1)))
                            pbar.update(3)
                            if return_stacked==True:
                                x.append(np.rot90(self.img_patch,  k=1, axes=(0, 1)))
                                y.append(np.rot90(self.lbl_patch,  k=1, axes=(0, 1)))
                                x.append(np.rot90(self.img_patch,  k=2, axes=(0, 1)))
                                y.append(np.rot90(self.lbl_patch,  k=2, axes=(0, 1)))
                                x.append(np.rot90(self.img_patch,  k=3, axes=(0, 1)))
                                y.append(np.rot90(self.lbl_patch,  k=3, axes=(0, 1)))
                        index+=1
                       
                print("#######################################################################################")
                print(f'{pbar.total} patches are saved as ".npy" format in "{os.getcwd()}\{folder_name}"')
                print("#######################################################################################")
        if return_stacked==True:
            self.patch_stacked=np.array(x, dtype="float32") 
            self.label_stacked=np.array(y, dtype="int")
            print("  ")
            print("#######################################################################################")
            print (f'The shape of stacked patches and labels are: {self.patch_stacked.shape, self.label_stacked.shape}' )
            print("#######################################################################################")
        if save_stack==True:
            print("   ")
            print("#######################################################################################")
            np.save(folder_name+"/Patch_stacked_"+str(self.patch_size)+".npy", self.patch_stacked)
            np.save(folder_name+"/label_stacked_"+str(self.patch_size)+".npy", self.label_stacked)
            print (f'The shape of stacked patches and labels are: {self.patch_stacked.shape, self.label_stacked.shape}' )
            print(f' Stacked patches and labels are saved as ".npy" format in: \n'
                  f'"{os.getcwd()}\{folder_name}"')
            print("#######################################################################################")

            
        if return_stacked==True:
            return self.patch_stacked, self.label_stacked
        
    def visualize(self,folder_name="npy",patches_to_show=1, band_num=1, fig_size=(10,20), dpi=96):
        """
        Using this function generated patches can be displayed
        
        parameters:
        -----------
        folder_name:        String
                            The folder name that passed for saving patches should be passed here 
                            to retrieve patches and display. 
        
        patches_to_show:    Integer
                            Number of random patches to display.
                            
        band_num:           Integer
                            the image band to be displayed.
                            to display the last band number just pass 0 and for the first band 1.
                            
        figsize:            Tuple 
                            plotting size for patches, the default is (10,20).
                            
        dpi:                Integer
                            Dots per inches (dpi) to determine the size of the figure in inches, the default value is 96.
        """

        print('image and label for the following patches will be displayed are: ')
        patch_dir=glob.glob(os.path.join(os.getcwd(),folder_name, "patch/*"))
        label_dir=glob.glob(os.path.join(os.getcwd(),folder_name, "label/*"))
  
        idx=np.random.randint(1, len(patch_dir), patches_to_show)
        if patch_dir[0][-3:]=="tif":
            fig,ax= plt.subplots(len(idx),2, figsize=fig_size, dpi=dpi)
            for i in range(len(idx)):
                file=patch_dir[idx[i]]
                print(file)
                file_=label_dir[idx[i]]
                print(file_)
                img=rs.open(file)
                img=img.read()
                lbl=rs.open(file_)
                lbl=lbl.read(1)
                if patches_to_show<2:
                    ax[i].imshow(img[band_num,:,:])
                    ax[i+1].imshow(lbl)
                else:
                    ax[i,0].imshow(img[band_num,:,:])
                    ax[i,1].imshow(lbl)
                    
        if patch_dir[0][-3:]=="npy":
            fig,ax= plt.subplots(len(idx),2, figsize=fig_size, dpi=dpi)
            for i in range(len(idx)):
                file=patch_dir[idx[i]]
                print(file)
                file_=label_dir[idx[i]]
                print(file_)
                img=np.load(file)
                lbl=np.load(file_)
                if patches_to_show<2:
                    ax[i].imshow(img[:,:,band_num])
                    ax[i+1].imshow(lbl)
                else:
                    ax[i,0].imshow(img[:,:,band_num])
                    ax[i,1].imshow(lbl)
       
                
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                