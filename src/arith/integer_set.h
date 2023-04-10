/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file integer_set.h
 * \brief Integer set based on MLIR Presburger set
 */
#ifndef TVM_ARITH_INTEGER_SET_H_
#define TVM_ARITH_INTEGER_SET_H_

#include <mlir/Analysis/Presburger/PresburgerRelation.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Analysis/Presburger/Simplex.h>

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

#include <limits>

#include "const_fold.h"

namespace tvm {
namespace arith {

using namespace mlir;
using namespace presburger;

// Acknowledgement: IntegerSet is based on Presburger set of MLIR.
/*!
 * \brief Symbolic interval set.
 *
 * \note IntegerSet provides compatible APIs with IntSet,
 *       and additional APIs that analyze multi-dimension interger set
 *       and related solver.
 */
class IntegerSetNode : public IntSetNode {
 public:
  PresburgerSet set;

  explicit IntegerSetNode(const PresburgerSet &set, const Array<Var> &vars)
    : set(std::move(set)), vars(vars) {};
  explicit IntegerSetNode()
    : set(PresburgerSet::getEmpty(PresburgerSpace::getSetSpace(0, 0, 0))) {}
  explicit IntegerSetNode(const PresburgerSet &set) : set(set) {}

  // visitor overload.
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("vars", &vars);
    if (auto* p = dynamic_cast<NodeAttrSetter*>(v)) {
      PrimExpr constraint = p->GetAttr("constraint");
      UpdateConstraint(constraint, vars);
    } else {
      PrimExpr constraint = GenerateConstraint();
      v->Visit("constraint", &constraint);
    }
  }

  void UpdateConstraint(const PrimExpr& constraint, const Array<Var>& vars);

  PrimExpr GenerateConstraint() const;

  void SetVars(const Array<Var> &new_vars) { vars = new_vars; }

  Array<Var> GetVars() const { return vars; }

  /*! \return whether integer set is empty */
  bool IsEmpty() const { return set.isIntegerEmpty(); }

  static constexpr const char* _type_key = "arith.IntegerSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(IntegerSetNode, IntSetNode);

 private:
  Array<Var> vars;
};

/*!
 * \brief Integer set used for multi-dimension integer analysis.
 * \sa IntegerSetNode
 */
class IntegerSet : public IntSet {
 public:
  /*!
   * \brief Make a new instance of integer set.
   * \param constraint The constraint to construct the set.
   * \param vars The variances that constraint describes about.
   * \return The created set.
   */
  TVM_DLL IntegerSet(const PrimExpr& constraint, const Array<Var>& vars);

  /*!
   * \brief Make a new instance of integer set, collect all vars as space vars.
   * \param constraint The constraint to construct the set.
   * \return The created set.
   */
  TVM_DLL IntegerSet(const PrimExpr& constraint);

  /*!
   * \brief Make a new instance of integer set.
   * \param set The PresburgurSet to construct the IntegerSet.
   * \param vars The variances that integer set describes about.
   * \return The created set.
   */
  TVM_DLL IntegerSet(const PresburgerSet &set, const Array<Var>& vars);

  TVM_DEFINE_OBJECT_REF_COW_METHOD(IntegerSetNode);
  TVM_DEFINE_OBJECT_REF_METHODS(IntegerSet, IntSet, IntegerSetNode);
};

/*!
 * \brief Create a union set of all sets
 * \param sets The sets to be combined
 * \return the set after union
 */
IntegerSet Union(const Array<IntegerSet>& sets);

/*!
 * \brief Create an intersected set of all sets
 * \param sets The sets to be intersected
 * \return the set after intersected
 */
IntegerSet Intersect(const Array<IntegerSet>& sets);

IntSet EvalSet(const PrimExpr& e, const IntegerSet& set);

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_INTEGER_SET_H_
