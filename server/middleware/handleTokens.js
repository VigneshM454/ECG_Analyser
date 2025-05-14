const jwt = require("jsonwebtoken");

function getCookieOptions(req){
  const isProduction = process.env.NODE_ENV === 'production';
  
  // Get the origin from the request headers to adapt to different frontends
  const origin = req.headers.origin;
  const isNetlify = origin && origin.includes('netlify');
  const cookieOptions = {
    httpOnly: true,
    secure: isProduction || isNetlify, // Must be true for cross-domain cookies
    sameSite: isNetlify ? 'none' : 'lax', // Must be 'none' for cross-domain cookies
    path: '/',
  };
  return cookieOptions;
}

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
        ...getCookieOptions(req),
        maxAge: 5 * 60 * 1000
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

function generateToken(req,res,email){
  const accessToken=jwt.sign({email:email},process.env.ACCESS_TOKEN, {expiresIn:'3m'});
    const refreshToken=jwt.sign({email:email},process.env.REFRESH_TOKEN, {expiresIn:'15m'});
    res.cookie("accessToken", accessToken, {
      ...getCookieOptions(req),
      maxAge: 5 * 60 * 1000, 
    });
    
    res.cookie('refreshToken', refreshToken, {
      ...getCookieOptions(req),
      maxAge: 24 * 60 * 60 * 1000, // 1 day in ms
    });
    
    console.log('cookies is ');
    console.log(res.getHeaders()['set-cookie']);
    // return res.send({status:200,msg:'success'})

}

module.exports = {validateToken,generateToken,getCookieOptions};
