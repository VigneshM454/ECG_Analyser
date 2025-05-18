
import Component from '@glimmer/component';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';
export default class EcgResultComponent extends Component {

  @tracked ecgResult={};
  colors = ['red','green','blue'];
  constructor(){
    super(...arguments)
    console.log(this.args.data);
    this.ecgResult={...this.args.data};
    console.log(this.ecgResult);
  }

  @action toPercentage(num) {
    return parseFloat((num * 100).toFixed(2));
  }

  // get limeFeatures(){
  //   const support=[];
  //   const contrast=[];
  //   for(var i=0;i<this.ecgResult.lime_explanation.length;i++){

  //   }
  //   // lime_explanation
  // }

  // getImage(){
  //   return this.ecgResult
  // }

}
