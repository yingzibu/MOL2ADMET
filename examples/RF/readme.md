Code: https://colab.research.google.com/drive/1WQzKw1zBqExVUyZvg8QsWJ4cJmJFL__U#scrollTo=jL3N2g7Q7IrX

 



Random Forest Evaluation: 


name |  	       mae      |       rmse      |       r2      |       spearman      
---|---|---|---|--- 
Caco2_Wang  |0.340&plusmn;0.001  |0.447&plusmn;0.001  |0.686&plusmn;0.002  |0.796&plusmn;0.001  
Lipophilicity_AstraZeneca |0.632&plusmn;0.001  |0.824&plusmn;0.001  |0.541&plusmn;0.001  |0.705&plusmn;0.001  
HydrationFreeEnergy_FreeSolv|0.724&plusmn;0.003  |1.271&plusmn;0.005  |0.901&plusmn;0.001  |0.970&plusmn;0.001  
Solubility_AqSolDB |0.903&plusmn;0.001  |1.276&plusmn;0.001  |0.700&plusmn;0.000  |0.834&plusmn;0.000  
LD50_Zhu	|0.426&plusmn;0.001  |0.581&plusmn;0.001  |0.623&plusmn;0.002  |0.746&plusmn;0.002 
Kp 	|5.956&plusmn;0.115  |12.234&plusmn;0.088  |0.207&plusmn;0.011  |0.438&plusmn;0.016  
Half_Life_Obach 	|29.358&plusmn;0.965  |105.727&plusmn;2.015  |0.220&plusmn;0.030  |0.326&plusmn;0.030  
Clearance_Hepatocyte_AZ 	|35.924&plusmn;0.245  |48.224&plusmn;0.145  |0.068&plusmn;0.006  |0.396&plusmn;0.011  
Clearance_Microsome_AZ 	|29.818&plusmn;0.205  |40.895&plusmn;0.290  |0.116&plusmn;0.013  |0.506&plusmn;0.015  
PPBR_AZ 	|8.667&plusmn;0.039  |12.998&plusmn;0.043  |0.303&plusmn;0.005  |0.536&plusmn;0.001 
VDss_Lombardo 	|3.132&plusmn;0.013  |5.643&plusmn;0.021  |0.318&plusmn;0.005  |0.570&plusmn;0.009 

     


name |       acc      |       auc      |       ap      
---|---|---|---  
CYP2C19_Veith |0.780&plusmn;0.003  |0.854&plusmn;0.000  |0.808&plusmn;0.001
CYP2D6_Veith |0.860&plusmn;0.001  |0.857&plusmn;0.001  |0.639&plusmn;0.000  
 CYP3A4_Veith  |0.767&plusmn;0.001  |0.849&plusmn;0.000  |0.790&plusmn;0.001  
CYP1A2_Veith |0.840&plusmn;0.001  |0.919&plusmn;0.000  |0.912&plusmn;0.001  
CYP2C9_Veith  	|0.796&plusmn;0.003  |0.867&plusmn;0.002  |0.737&plusmn;0.002  
BBB_Martins 	|0.856&plusmn;0.005  |0.904&plusmn;0.004  |0.960&plusmn;0.004  
Bioavailability_Ma 	|0.747&plusmn;0.007  |0.718&plusmn;0.007  |0.837&plusmn;0.003  
 Pgp_Broccatelli 	|0.831&plusmn;0.011  |0.906&plusmn;0.003  |0.933&plusmn;0.002  
HIA_Hou 	|0.963&plusmn;0.004  |0.979&plusmn;0.002  |0.996&plusmn;0.001  
 PAMPA_NCATS  	|0.857&plusmn;0.008  |0.797&plusmn;0.001  |0.950&plusmn;0.001  
hERG_Karim 	|0.836&plusmn;0.003  |0.905&plusmn;0.000  |0.908&plusmn;0.000  
 AMES	|0.835&plusmn;0.002  |0.900&plusmn;0.001  |0.909&plusmn;0.001  
 CYP2C9_Substrate_CarbonMangels 	|0.769&plusmn;0.006  |0.600&plusmn;0.002  |0.263&plusmn;0.001  
 CYP2D6_Substrate_CarbonMangels 	|0.742&plusmn;0.004  |0.791&plusmn;0.006  |0.539&plusmn;0.030  
CYP3A4_Substrate_CarbonMangels	|0.624&plusmn;0.009  |0.657&plusmn;0.000  |0.627&plusmn;0.003  
DILI 	|0.818&plusmn;0.010  |0.886&plusmn;0.003  |0.882&plusmn;0.004  
Skin Reaction 	|0.831&plusmn;0.012  |0.839&plusmn;0.004  |0.828&plusmn;0.004  
Carcinogens_Lagunin 	|0.893&plusmn;0.000  |0.820&plusmn;0.015  |0.744&plusmn;0.006  
ClinTox  	|0.921&plusmn;0.002  |0.904&plusmn;0.006  |0.455&plusmn;0.008  
 
 
