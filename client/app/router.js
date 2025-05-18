import EmberRouter from '@ember/routing/router';
import config from 'client/config/environment';

export default class Router extends EmberRouter {
  location = config.locationType;
  rootURL = config.rootURL;
}

Router.map(function () {
  this.route('home');
  this.route('result');
  this.route('upload');
  this.route('patient', { path: '/patient/:id' });
  this.route('record', { path: '/record/:id' });
  this.route('upload-existing', { path: '/upload-existing/:id' });
});
