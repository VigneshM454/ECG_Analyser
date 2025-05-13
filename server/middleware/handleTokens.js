const jwt = require("jsonwebtoken");

function validateToken(req, res, next) {
  console.log('entered validateToken');
  console.log(req.cookies);
  const accessToken = req.cookies.accessToken;
  const refreshToken = req.cookies.refreshToken;

  // No tokens present
  if (!accessToken && !refreshToken) return res.sendStatus(401);

  // Try access token first
  jwt.verify(accessToken, process.env.ACCESS_TOKEN, (err, user) => {
    if (!err) {
      console.log(user);
      req.user = user;
      return next(); // valid access token
    }

    // If access token failed, check refresh token
    if (!refreshToken) return res.sendStatus(401);

    jwt.verify(refreshToken, process.env.REFRESH_TOKEN, (err, user) => {
      if (err) return res.sendStatus(403); // Refresh token invalid

      // Issue a new access token
      const newAccessToken = jwt.sign({ email: user.email }, process.env.ACCESS_TOKEN, { expiresIn: "3m" });
      res.cookie("accessToken", newAccessToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: "strict",
        maxAge: 5 * 60 * 1000, 
        path: '/', 
      });
      // res.cookie("accessToken", newAccessToken, {
      //   httpOnly: true,
      //   secure: true,
      //   sameSite: "strict",
      //   maxAge: 3 * 60 * 1000, // 3 minutes
      // });

      req.user = user;
      next();
    });
  });
}

function generateToken(res,email){
  const accessToken=jwt.sign({email:email},process.env.ACCESS_TOKEN, {expiresIn:'3m'});
    const refreshToken=jwt.sign({email:email},process.env.REFRESH_TOKEN, {expiresIn:'15m'});
    res.cookie("accessToken", accessToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: "strict",
      maxAge: 5 * 60 * 1000, 
      path: '/', 
    });
    
    res.cookie('refreshToken', refreshToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 24 * 60 * 60 * 1000, // 1 day in ms
      path: '/',
    });
    
    console.log('cookies is ');
    console.log(res.getHeaders()['set-cookie']);
    // return res.send({status:200,msg:'success'})

}

module.exports = {validateToken,generateToken};
