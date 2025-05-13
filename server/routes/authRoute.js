require('dotenv').config();
const express=require('express');
const axios=require('axios')
const jwt=require('jsonwebtoken');
const userModel=require('../models/userModel')
const {comparePassword,hashPassword} =require('../securePassword')
const {validateToken,generateToken} = require('../middleware/handleTokens')

const authRoute=express.Router();

authRoute.post('/createAccount',async(req,res,next)=>{
    console.log(req.body)
    const {name,email,pass,phone}=req.body
    const password= hashPassword(pass)
    console.log(password);

    const user=await userModel.findOne({email:email})
    console.log((user?'user is present':'user is not present '));
    
    if(user) return res.status(300).send({msg:'Account with same email already exist'})

    const newUser=await userModel.create({name,email,phone,password})
    console.log(newUser)
    
    await generateToken(res,email)
    return res.send({status:200,msg:'Account created successfully',name:newUser.name,email:newUser.email,phone:newUser.phone})
 
})

authRoute.get('/isAuthentic',validateToken,async(req,res,next)=>{
    console.log('from isAuthentic route');
    // console.log(req.user);
    const user= await userModel.findOne({email:req.user.email})
    // console.log(user);
    const userData={name:user.name,email:user.email,phone:user.phone}
    return res.send({status:200,msg:'Authenctic ',userData})
})

authRoute.post('/login',async(req,res,next)=>{
    console.log('inside login');
    console.log(req.body);
    const {email,pass}=req.body
    const user= await userModel.findOne({email:email})
    if(!user) return res.status(300).send({msg:'No such user exist'})
    const isValidPass=comparePassword(pass,user.password)
    if(! isValidPass) return res.status(400).send({msg:'Invalid credentials'})
    console.log(user);

    await generateToken(res,email)

    res.status(200).send({msg:'Login succcess', name:user.name,email:user.email,phone:user.phone})
})

authRoute.post('/logout',async(req,res,next)=>{
    console.log('inside logout');
    res.clearCookie("accessToken", {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production', 
        sameSite: process.env.NODE_ENV === 'production' ? 'strict' : 'lax',
        // secure: process.env.NODE_ENV === 'production',
        // sameSite: "strict",
         path: '/', 
    });
    
    res.clearCookie("refreshToken", {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production', 
        sameSite: process.env.NODE_ENV === 'production' ? 'strict' : 'lax',
        // secure: process.env.NODE_ENV === 'production',
        // sameSite: "strict",
        path: '/', 
    });
    
    return res.status(200).json({ msg: "Logged out successfully" });
})



module.exports=authRoute;

    // const accessToken=jwt.sign({email:email},process.env.ACCESS_TOKEN, {expiresIn:'3m'});
    // const refreshToken=jwt.sign({email:email},process.env.REFRESH_TOKEN, {expiresIn:'15m'});
    // res.cookie('accessToken',accessToken,{
    //     httpOnly:true,secure:true,sameSite:'strict',maxAge:60000
    // });
    // res.cookie('refreshToken',refreshToken, {
    //     httpOnly:true,secure:true,sameSite:'strict',maxAge:300000
    // });
    // console.log('cookies is ');


// adminRoutes.post('/admin-login',async(req,res,next)=>{
//     const {email,passwd}=req.body;
//     console.log(req.body);
//     if(email==='' || passwd==='')return res.send({status:400,msg:"Input fields are missing"})
//     const admindata= await adminModel.findOne({email:email});
//     console.log(admindata);
//     if(admindata==null) return res.send({status:404,msg:'So such admin exist'});
//     if(!comparePassword(passwd,admindata.passwd)) return res.send({status:300,msg:'Invalid credentials'});
//     //create token

//     const accessToken=jwt.sign({email:email},process.env.ACCESS_TOKEN, {expiresIn:'3m'});
//     const refreshToken=jwt.sign({email:email},process.env.REFRESH_TOKEN, {expiresIn:'15m'});
//     res.cookie('accessToken',accessToken,{
//         httpOnly:true,secure:true,sameSite:'strict',maxAge:60000
//     });
//     res.cookie('refreshToken',refreshToken, {
//         httpOnly:true,secure:true,sameSite:'strict',maxAge:300000
//     });
//     console.log('cookies is ');
//     return res.send({status:200,msg:'success'})
// })

// adminRoutes.post('/admin-logout',verifyToken,(req,res,next)=>{
//     delete req.email;
//     res.clearCookie('accessToken');
//     res.clearCookie('refreshToken')
//     return res.send({status:200,msg:'Logout success'});
// })

// adminRoutes.post('/admin-edit',verifyToken,async(req,res,next)=>{
//     console.log('in /admin-edit');
//     console.log(req.body);
//     const {fname,lname,email,passwd}=req.body;
//     if(fname===''||lname===''||email===''||passwd==='') return res.send({status:400,msg:'Some input fields are missing'})
//     const passwd2=hashPassword(passwd)
//     console.log(passwd2);
//     try{
//         await adminModel.findOneAndUpdate({email:email},{$set:{
//             fname,lname,email,passwd:passwd2
//         }});
//         const adminDetails=await adminModel.findOne({email:email})
//         return res.send({status:200,msg:'Profile info updated successfully!',data:adminDetails})         
//     }catch(err){
//         console.log(err);
//         return res.send({status:300,msg:err})
//     }
// })

// adminRoutes.get('/get-admindata',verifyToken,async(req,res,next)=>{
//     console.log('entered get-admindata');
//     const email=req.email
//     delete req.email;
//     console.log(email);
//     const adminData=await adminModel.findOne({email:email});
//     //if adminData is empty
//     if(adminData===null) return res.send({status:404,msg:'No such user in database'})
//     console.log('admin data is ');
//     console.log(adminData);
//     return res.send({status:200,msg:'authorized',data:adminData})
// })


// const adminModel=require('../models/adminModel');
// const verifyToken=require('../auth/adminAuth')

// authRoute.get('/hi',async(req,res,next)=>{
//     console.log('entered hi route');
//     res.send('Hi dude')
    
// })


    // axios.post('http://127.0.0.1:5000/demo',JSON.stringify(newUser),
    // {headers: {
    //     'Content-Type': 'application/json',  // Specify content type as JSON
    //   },}
    // )
    // .then(res=>res.data)
    // .then(data=>{
    //     console.log(data);
    // })
    // .catch(err=>{
    //     console.log(err);
    //     console.log('no response from flask server')
    // })
    
    // console.log(newUser);
    // console.log(user._id?true:false);
   